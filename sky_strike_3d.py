#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKY STRIKE 3D  -  3D fighter-jet combat game (pure Python + pygame).

A small software 3D engine (no OpenGL needed), so it also runs in Termux on
Android through Termux:X11.

Keyboard
  W / S or Up / Down : pitch (nose up / down)      A / D or Left / Right : bank + turn
  Q / E : rudder       Shift : afterburner         Ctrl / X : brake
  SPACE / J : machine gun       M / K : homing missile (lock an enemy first)
  C : camera (chase / cockpit / far)    T : auto-fire      I : invert pitch
  P : pause   F : fullscreen   ESC : quit   ENTER : start / restart

Touch
  Drag on the LEFT half of the screen = flight stick, buttons on the right.

Options
  --quality low|med|high   (960x540 / 1280x720 / 1920x1080 Full HD)
  --hd  same as --quality high     --fullscreen   --mute   --fps
"""
import argparse
import array
import math
import os
import random
import sys
from operator import itemgetter

import pygame

TILE = 500.0
NT = 32                     # tiles per side of the (wrapping) world
TW = TILE * NT
WATER = -70.0
HAZE = (190, 208, 226)
SKY_TOP = (36, 90, 188)


# --------------------------------------------------------------------------
# small vector helpers (tuples)
# --------------------------------------------------------------------------
def clamp(v, a, b):
    return a if v < a else (b if v > b else v)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(v):
    l = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if l < 1e-9:
        return (0.0, 0.0, 1.0)
    return (v[0] / l, v[1] / l, v[2] / l)


def lc(a, ca, b, cb):
    return (a[0] * ca + b[0] * cb, a[1] * ca + b[1] * cb, a[2] * ca + b[2] * cb)


def mix(c, h, t):
    return (int(c[0] + (h[0] - c[0]) * t), int(c[1] + (h[1] - c[1]) * t), int(c[2] + (h[2] - c[2]) * t))


def ortho(f, u):
    f = norm(f)
    r = norm(cross(u, f))
    u = cross(f, r)
    return f, u, r


LIGHT = norm((0.45, 0.8, 0.35))      # direction towards the sun


def rot_y(v, a):
    c, s = math.cos(a), math.sin(a)
    return (v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c)


# body rotations (basis f=forward, u=up, r=right)
def b_yaw(b, a):
    c, s = math.cos(a), math.sin(a)
    b.f, b.r = lc(b.f, c, b.r, s), lc(b.r, c, b.f, -s)


def b_pitch(b, a):          # nose up positive
    c, s = math.cos(a), math.sin(a)
    b.f, b.u = lc(b.f, c, b.u, s), lc(b.u, c, b.f, -s)


def b_roll(b, a):           # right wing down positive
    c, s = math.cos(a), math.sin(a)
    b.r, b.u = lc(b.r, c, b.u, -s), lc(b.u, c, b.r, s)


def steer(b, d, rate, dt, maxbank=1.1):
    """Turn body b towards unit direction d (limited turn rate) and bank nicely."""
    f = b.f
    ang = math.acos(clamp(dot(f, d), -1.0, 1.0))
    if ang > 1e-4:
        k = min(1.0, rate * dt / ang)
        nf = norm(lc(f, 1 - k, d, k))
    else:
        nf = f
    rl_old = norm(cross((0, 1, 0), f)) if abs(f[1]) < 0.97 else b.r
    lat = dot(d, rl_old)
    tb = clamp(lat * 2.5, -1.0, 1.0) * maxbank
    b.bank += (tb - b.bank) * min(1.0, 3.5 * dt)
    rl = norm(cross((0, 1, 0), nf)) if abs(nf[1]) < 0.97 else rl_old
    ul = cross(nf, rl)
    cb, sb = math.cos(b.bank), math.sin(b.bank)
    b.f = nf
    b.r = lc(rl, cb, ul, -sb)
    b.u = lc(ul, cb, rl, sb)


def seg_hit(p0, p1, c, r):
    dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    l2 = dx * dx + dy * dy + dz * dz
    t = 0.0
    if l2 > 1e-9:
        t = clamp(((c[0] - p0[0]) * dx + (c[1] - p0[1]) * dy + (c[2] - p0[2]) * dz) / l2, 0.0, 1.0)
    qx, qy, qz = p0[0] + dx * t - c[0], p0[1] + dy * t - c[1], p0[2] + dz * t - c[2]
    return qx * qx + qy * qy + qz * qz < r * r


# --------------------------------------------------------------------------
# meshes
# --------------------------------------------------------------------------
class Mesh:
    def __init__(self):
        self.v = []
        self.faces = []
        self.radius = 1.0

    def vert(self, x, y, z):
        self.v.append((x, y, z))
        return len(self.v) - 1

    def face(self, idx, color, two=False, ctr=(0.0, 0.0, 0.0)):
        a, b, c = self.v[idx[0]], self.v[idx[1]], self.v[idx[2]]
        n = cross((b[0] - a[0], b[1] - a[1], b[2] - a[2]), (c[0] - a[0], c[1] - a[1], c[2] - a[2]))
        l = math.sqrt(dot(n, n))
        if l < 1e-9:
            return
        n = (n[0] / l, n[1] / l, n[2] / l)
        if not two:
            k = len(idx)
            fc = (sum(self.v[i][0] for i in idx) / k, sum(self.v[i][1] for i in idx) / k,
                  sum(self.v[i][2] for i in idx) / k)
            if dot(n, (fc[0] - ctr[0], fc[1] - ctr[1], fc[2] - ctr[2])) < 0:
                n = (-n[0], -n[1], -n[2])
        self.faces.append((tuple(idx), color, n[0], n[1], n[2], dot(n, a), two))

    def face_axis(self, idx, color, ax, az):
        """closed solid of revolution around a vertical axis at (ax, az)"""
        fy = sum(self.v[i][1] for i in idx) / len(idx)
        self.face(idx, color, False, (ax, fy, az))

    def finish(self):
        self.radius = max(math.sqrt(dot(v, v)) for v in self.v) + 1.0
        return self


def make_jet(body, wing, accent, s=2.0):
    m = Mesh()

    def v(x, y, z):
        return m.vert(x * s, y * s, z * s)

    dark = tuple(int(c * 0.62) for c in body)
    nose = v(0, 0, 9.5)
    a = [v(0, 1.1, 2), v(1.1, 0, 2), v(0, -0.8, 2), v(-1.1, 0, 2)]
    b = [v(0, 0.9, -7), v(0.9, 0, -7), v(0, -0.7, -7), v(-0.9, 0, -7)]
    cols = [body, dark, dark, body]
    for i in range(4):
        j = (i + 1) % 4
        m.face((nose, a[i], a[j]), cols[i])
        m.face((a[i], a[j], b[j], b[i]), cols[i])
    m.face((b[0], b[1], b[2], b[3]), (255, 150, 40))      # exhaust
    for sx in (-1, 1):
        w0, w1, w2 = v(sx * 1.0, 0, 3.5), v(sx * 7.6, -0.3, -5.2), v(sx * 1.0, 0, -5.2)
        m.face((w0, w1, w2), wing, True)
        t0, t1, t2 = v(sx * 0.9, 0, -5.4), v(sx * 3.9, 0, -8.4), v(sx * 0.9, 0, -7.4)
        m.face((t0, t1, t2), wing, True)
        f0, f1, f2 = v(sx * 0.5, 0.9, -3.4), v(sx * 1.4, 3.3, -7.6), v(sx * 0.5, 0.9, -7.3)
        m.face((f0, f1, f2), accent, True)
        tip0, tip1, tip2 = v(sx * 6.4, -0.25, -3.6), v(sx * 7.6, -0.3, -5.2), v(sx * 6.0, -0.25, -5.0)
        m.face((tip0, tip1, tip2), accent, True)
    c0, c1, c2, c3 = v(-0.65, 0.95, 3.4), v(0.65, 0.95, 3.4), v(0, 1.0, -0.4), v(0, 1.95, 1.5)
    ctr = (0, 1.2 * s, 1.4 * s)
    for tri in ((c0, c1, c3), (c1, c2, c3), (c2, c0, c3)):
        m.face(tri, (90, 175, 240), False, ctr)
    return m.finish()


def make_missile():
    m = Mesh()
    nose = m.vert(0, 0, 9)
    r1 = [m.vert(1.3, 0, 4), m.vert(0, 1.3, 4), m.vert(-1.3, 0, 4), m.vert(0, -1.3, 4)]
    r2 = [m.vert(1.3, 0, -8), m.vert(0, 1.3, -8), m.vert(-1.3, 0, -8), m.vert(0, -1.3, -8)]
    for i in range(4):
        j = (i + 1) % 4
        m.face((nose, r1[i], r1[j]), (220, 60, 50))
        m.face((r1[i], r1[j], r2[j], r2[i]), (235, 235, 240))
    for (x, y) in ((4.2, 0), (-4.2, 0), (0, 4.2), (0, -4.2)):
        a = m.vert(x * 0.3, y * 0.3, -3)
        b = m.vert(x, y, -8.5)
        c = m.vert(x * 0.3, y * 0.3, -8.2)
        m.face((a, b, c), (180, 185, 195), True)
    return m.finish()


def make_mountain(rng, R, H):
    m = Mesh()
    k = 8
    base, mid = [], []
    angs = [2 * math.pi * i / k for i in range(k)]
    for a in angs:
        rr = R * rng.uniform(0.85, 1.15)
        base.append(m.vert(math.cos(a) * rr, -40, math.sin(a) * rr))
    for a in angs:
        rr = R * rng.uniform(0.34, 0.5)
        mid.append(m.vert(math.cos(a) * rr, H * rng.uniform(0.5, 0.64), math.sin(a) * rr))
    peak = m.vert(rng.uniform(-0.08, 0.08) * R, H, rng.uniform(-0.08, 0.08) * R)
    snow = H > 650
    for i in range(k):
        j = (i + 1) % k
        g = rng.randint(-14, 14)
        low = (104 + g, 100 + g, 88 + g) if i % 2 else (92 + g, 104 + g, 80 + g)
        m.face_axis((base[i], base[j], mid[j], mid[i]), low, 0, 0)
        top = (240 - abs(g), 243 - abs(g), 248 - abs(g)) if snow else (132 + g, 124 + g, 112 + g)
        m.face_axis((mid[i], mid[j], peak), top, 0, 0)
    return m.finish()


def make_city(rng):
    m = Mesh()
    for _ in range(rng.randint(4, 8)):
        cx, cz = rng.uniform(-110, 110), rng.uniform(-110, 110)
        w, d, h = rng.uniform(26, 56), rng.uniform(26, 56), rng.uniform(40, 170)
        g = rng.randint(-18, 18)
        col = rng.choice([(170, 172, 180), (150, 160, 178), (190, 180, 165), (140, 140, 150)])
        col = tuple(clamp(c + g, 0, 255) for c in col)
        x0, x1, z0, z1 = cx - w / 2, cx + w / 2, cz - d / 2, cz + d / 2
        vs = [m.vert(x, y, z) for y in (0, h) for z in (z0, z1) for x in (x0, x1)]
        ctr = (cx, h / 2, cz)
        m.face((vs[4], vs[5], vs[7], vs[6]), tuple(min(255, int(c * 1.15)) for c in col), False, ctr)
        for q, sh in (((0, 1, 5, 4), 0.85), ((1, 3, 7, 5), 0.7), ((3, 2, 6, 7), 0.8), ((2, 0, 4, 6), 0.65)):
            m.face(tuple(vs[i] for i in q), tuple(int(c * sh) for c in col), False, ctr)
    return m.finish()


def make_grove(rng):
    m = Mesh()
    for _ in range(rng.randint(4, 9)):
        cx, cz = rng.uniform(-120, 120), rng.uniform(-120, 120)
        h, r = rng.uniform(45, 80), rng.uniform(16, 26)
        g = rng.randint(-12, 12)
        ring = [m.vert(cx + math.cos(a) * r, 0, cz + math.sin(a) * r)
                for a in (0, math.pi / 2, math.pi, 3 * math.pi / 2)]
        apex = m.vert(cx, h, cz)
        for i in range(4):
            m.face_axis((ring[i], ring[(i + 1) % 4], apex), (28 + g, 92 + g, 44 + g), cx, cz)
    return m.finish()


# --------------------------------------------------------------------------
# world: terrain + static scenery
# --------------------------------------------------------------------------
class World:
    def __init__(self, seed=3):
        rng = random.Random(seed)
        N = NT
        H = [[0.0] * N for _ in range(N)]
        for i in range(N):
            a = 2 * math.pi * i / N
            for j in range(N):
                b = 2 * math.pi * j / N
                h = (170 * math.sin(3 * a + 1.0) * math.cos(2 * b + 0.5)
                     + 95 * math.sin(5 * a + 2.0) * math.sin(4 * b + 1.3)
                     + 55 * math.cos(8 * a + 3 * b + 0.7)
                     + 28 * math.sin(11 * a - 9 * b + 2.1))
                H[i][j] = max(WATER, h)
        self.H = H
        # tile colours (pre-lit)
        self.C = [[None] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                i1, j1 = (i + 1) % N, (j + 1) % N
                hs = (H[i][j], H[i1][j], H[i][j1], H[i1][j1])
                avg = sum(hs) / 4
                jit = ((i * 7 + j * 13) % 5 - 2) * 3
                if max(hs) <= WATER + 0.5:
                    col = (34 + jit, 98 + jit, 168 + jit)
                else:
                    dhx = ((H[i1][j] + H[i1][j1]) - (H[i][j] + H[i][j1])) / (2 * TILE)
                    dhz = ((H[i][j1] + H[i1][j1]) - (H[i][j] + H[i1][j])) / (2 * TILE)
                    n = norm((-dhx, 1.0, -dhz))
                    sh = clamp(0.62 + 0.75 * max(0.0, dot(n, LIGHT)) - 0.25, 0.6, 1.2)
                    if avg < WATER + 38:
                        base = (216, 198, 142)
                    elif avg < 90:
                        base = (72 + jit, 148 + jit, 62 + jit)
                    elif avg < 200:
                        base = (58 + jit, 118 + jit, 58 + jit)
                    elif avg < 290:
                        base = (128 + jit, 122 + jit, 108 + jit)
                    else:
                        base = (236, 238, 242)
                    col = tuple(int(clamp(c * sh, 0, 255)) for c in base)
                self.C[i][j] = col
        # scenery
        self.statics = {}       # (i,j) -> list of (kind, ox, oy, oz, mesh)
        self.mountains = {}     # (i,j) -> list of (ox, oz, R, H, base_y)
        self.mount_meshes = []
        for i in range(NT):
            for j in range(NT):
                r = random.Random(seed * 100003 + i * 131 + j)
                avg = (self.H[i][j] + self.H[(i + 1) % N][j] + self.H[i][(j + 1) % N]
                       + self.H[(i + 1) % N][(j + 1) % N]) / 4
                if avg <= WATER + 25:
                    continue
                roll = r.random()
                ox, oz = r.uniform(60, TILE - 60), r.uniform(60, TILE - 60)
                by = self.terrain(i * TILE + ox, j * TILE + oz)
                if roll < 0.075:
                    R = r.uniform(520, 1250)
                    Hh = r.uniform(430, 1250)
                    mesh = make_mountain(r, R, Hh)
                    self.statics.setdefault((i, j), []).append(("m", ox, by, oz, mesh))
                    self.mountains.setdefault((i, j), []).append((ox, oz, R, Hh, by))
                elif roll < 0.115 and -20 < avg < 110:
                    self.statics.setdefault((i, j), []).append(("c", ox, by, oz, make_city(r)))
                elif roll < 0.33 and -10 < avg < 230:
                    self.statics.setdefault((i, j), []).append(("t", ox, by, oz, make_grove(r)))

    def terrain(self, x, z):
        gx, gz = x / TILE, z / TILE
        ix, iz = int(math.floor(gx)), int(math.floor(gz))
        fx, fz = gx - ix, gz - iz
        H = self.H
        i0, j0, i1, j1 = ix % NT, iz % NT, (ix + 1) % NT, (iz + 1) % NT
        a = H[i0][j0] * (1 - fx) + H[i1][j0] * fx
        b = H[i0][j1] * (1 - fx) + H[i1][j1] * fx
        return a * (1 - fz) + b * fz

    def ground_h(self, x, z):
        h = self.terrain(x, z)
        ix, iz = int(math.floor(x / TILE)), int(math.floor(z / TILE))
        mt = self.mountains
        for di in range(-3, 4):
            for dj in range(-3, 4):
                lst = mt.get(((ix + di) % NT, (iz + dj) % NT))
                if lst:
                    for (ox, oz, R, Hh, by) in lst:
                        dx = x - ((ix + di) * TILE + ox)
                        dz = z - ((iz + dj) * TILE + oz)
                        d2 = dx * dx + dz * dz
                        if d2 < R * R:
                            hh = by + Hh * (1.0 - math.sqrt(d2) / (R * 1.05))
                            if hh > h:
                                h = hh
        return h

    def color_at(self, x, z):
        return self.C[int(math.floor(x / TILE)) % NT][int(math.floor(z / TILE)) % NT]


# --------------------------------------------------------------------------
# 3D renderer (painter's algorithm, flat shading, haze)
# --------------------------------------------------------------------------
def clip_near(pts, near):
    out = []
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        ina, inb = a[2] >= near, b[2] >= near
        if ina:
            out.append(a)
        if ina != inb:
            t = (near - a[2]) / (b[2] - a[2])
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, near))
    return out


def clip_rect(pts, W, H):
    """Clip a polygon to the screen rectangle (pygame is very slow on huge polygons)."""
    out = list(pts)
    for axis, lim, less in ((0, -2.0, False), (0, W + 2.0, True), (1, -2.0, False), (1, H + 2.0, True)):
        if not out:
            break
        res = []
        n = len(out)
        for i in range(n):
            a, b = out[i], out[(i + 1) % n]
            ain = (a[axis] <= lim) if less else (a[axis] >= lim)
            bin_ = (b[axis] <= lim) if less else (b[axis] >= lim)
            if ain:
                res.append(a)
            if ain != bin_:
                t = (lim - a[axis]) / (b[axis] - a[axis])
                res.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        out = res
    return out


def is_big(pts, W, H):
    for q in pts:
        if q[0] < -300 or q[0] > W + 300 or q[1] < -300 or q[1] > H + 300:
            return True
    return False


class Cam:
    def __init__(self):
        self.pos = (0.0, 0.0, 0.0)
        self.f = (0.0, 0.0, 1.0)
        self.u = (0.0, 1.0, 0.0)
        self.r = (1.0, 0.0, 0.0)


class Scene3D:
    def __init__(self, surf, world, quality):
        self.surf = surf
        self.world = world
        self.W, self.H = surf.get_size()
        self.cx, self.cy = self.W / 2.0, self.H / 2.0
        self.F = self.W * 0.86
        self.tx = (self.W / 2.0) / self.F * 1.2
        self.ty = (self.H / 2.0) / self.F * 1.2
        self.near = 6.0
        self.R = {"low": 7, "med": 10, "high": 13}[quality]
        self.quality = quality
        self.far = self.R * TILE
        self.fog_s = self.far * 0.5
        self.fog_e = self.far * 0.97
        self.items = []
        self.cam = Cam()
        self.tree_range = {"low": 2, "med": 4, "high": 5}[quality]
        self.city_range = {"low": 4, "med": 6, "high": 8}[quality]

    # ---- sky ------------------------------------------------------------
    def draw_sky(self):
        cam, F, cx, cy = self.cam, self.F, self.cx, self.cy
        surf = self.surf
        surf.fill(HAZE)
        ur, uu, uf = cam.r[1], cam.u[1], cam.f[1]
        ln = math.hypot(ur, uu)
        B = 5000.0
        if ln < 1e-3:
            if uf > 0:
                surf.fill(SKY_TOP)
            return
        n = (ur / ln, uu / ln)
        c0 = -uf * F / ln
        t = (-n[1], n[0])

        def poly(d):
            p0 = (n[0] * c0, n[1] * c0)
            q = [(p0[0] + t[0] * B, p0[1] + t[1] * B),
                 (p0[0] - t[0] * B, p0[1] - t[1] * B),
                 (p0[0] - t[0] * B + n[0] * d, p0[1] - t[1] * B + n[1] * d),
                 (p0[0] + t[0] * B + n[0] * d, p0[1] + t[1] * B + n[1] * d)]
            return clip_rect([(cx + x, cy - y) for x, y in q], self.W, self.H)

        pl = poly(B)
        if len(pl) >= 3:
            pygame.draw.polygon(surf, SKY_TOP, pl)
        bands = (520, 400, 300, 220, 155, 100, 58, 24)
        for k, d in enumerate(bands):
            col = mix(SKY_TOP, HAZE, (k + 1) / (len(bands) + 0.6))
            pl = poly(d)
            if len(pl) >= 3:
                pygame.draw.polygon(surf, col, pl)
        # sun
        sx, sy, sz = dot(LIGHT, cam.r), dot(LIGHT, cam.u), dot(LIGHT, cam.f)
        if sz > 0.05:
            X, Y = cx + sx / sz * F, cy - sy / sz * F
            if -200 < X < self.W + 200 and -200 < Y < self.H + 200:
                pygame.draw.circle(surf, (225, 232, 240), (int(X), int(Y)), int(self.W * 0.05))
                pygame.draw.circle(surf, (245, 244, 220), (int(X), int(Y)), int(self.W * 0.032))
                pygame.draw.circle(surf, (255, 252, 235), (int(X), int(Y)), int(self.W * 0.02))

    # ---- ground ---------------------------------------------------------
    def draw_ground(self):
        cam, F, cx, cy = self.cam, self.F, self.cx, self.cy
        px, py, pz = cam.pos
        cr, cu, cf = cam.r, cam.u, cam.f
        R, near = self.R, self.near
        ci, cj = int(math.floor(px / TILE)), int(math.floor(pz / TILE))
        Hh = self.world.H
        Cm = self.world.C
        n = 2 * R + 2
        verts = []
        for gx in range(ci - R, ci + R + 2):
            wx = gx * TILE - px
            ar, au, af = wx * cr[0], wx * cu[0], wx * cf[0]
            hcol = Hh[gx % NT]
            col = []
            for gz in range(cj - R, cj + R + 2):
                wz = gz * TILE - pz
                dy = hcol[gz % NT] - py
                X = ar + wz * cr[2] + dy * cr[1]
                Y = au + wz * cu[2] + dy * cu[1]
                Z = af + wz * cf[2] + dy * cf[1]
                if Z >= near:
                    k = F / Z
                    col.append((X, Y, Z, cx + X * k, cy - Y * k))
                else:
                    col.append((X, Y, Z, None, None))
            verts.append(col)
        out = []
        W, H = self.W, self.H
        fs, fe = self.fog_s, self.fog_e
        inv = 1.0 / (fe - fs)
        for a in range(n - 1):
            c0, c1 = verts[a], verts[a + 1]
            Ccol = Cm[(ci - R + a) % NT]
            for b in range(n - 1):
                v00, v01, v11, v10 = c0[b], c0[b + 1], c1[b + 1], c1[b]
                z = (v00[2] + v01[2] + v11[2] + v10[2]) * 0.25
                if z > fe:
                    continue
                s00, s01, s11, s10 = v00[3], v01[3], v11[3], v10[3]
                if s00 is not None and s01 is not None and s11 is not None and s10 is not None:
                    y00, y01, y11, y10 = v00[4], v01[4], v11[4], v10[4]
                    if (s00 < 0 and s01 < 0 and s11 < 0 and s10 < 0) or \
                       (s00 > W and s01 > W and s11 > W and s10 > W) or \
                       (y00 < 0 and y01 < 0 and y11 < 0 and y10 < 0) or \
                       (y00 > H and y01 > H and y11 > H and y10 > H):
                        continue
                    pts = ((s00, y00), (s01, y01), (s11, y11), (s10, y10))
                else:
                    if v00[2] < near and v01[2] < near and v11[2] < near and v10[2] < near:
                        continue
                    poly = clip_near([v00[:3], v01[:3], v11[:3], v10[:3]], near)
                    if len(poly) < 3:
                        continue
                    pts = [(cx + X * F / Z, cy - Y * F / Z) for X, Y, Z in poly]
                if is_big(pts, W, H):
                    pts = clip_rect(pts, W, H)
                    if len(pts) < 3:
                        continue
                col = Ccol[(cj - R + b) % NT]
                if z > fs:
                    col = mix(col, HAZE, (z - fs) * inv)
                out.append((z, col, pts))
        out.sort(key=itemgetter(0), reverse=True)
        poly_draw = pygame.draw.polygon
        surf = self.surf
        for z, col, pts in out:
            poly_draw(surf, col, pts)

    def draw_shadow(self, x, z, fx, fz, size=1.0):
        w = self.world
        gh = w.ground_h(x, z) + 2.0
        l = math.hypot(fx, fz) or 1.0
        fx, fz = fx / l, fz / l
        rx, rz = fz, -fx
        col = tuple(int(c * 0.55) for c in w.color_at(x, z))
        pts3 = [(x + fx * 24 * size, gh, z + fz * 24 * size),
                (x + rx * 26 * size, gh, z + rz * 26 * size),
                (x - fx * 20 * size, gh, z - fz * 20 * size),
                (x - rx * 26 * size, gh, z - rz * 26 * size)]
        cam = self.cam
        cv = []
        for (X, Y, Z) in pts3:
            dx, dy, dz = X - cam.pos[0], Y - cam.pos[1], Z - cam.pos[2]
            cz = dx * cam.f[0] + dy * cam.f[1] + dz * cam.f[2]
            if cz < self.near:
                return
            cv.append((dx * cam.r[0] + dy * cam.r[1] + dz * cam.r[2],
                       dx * cam.u[0] + dy * cam.u[1] + dz * cam.u[2], cz))
        pts = [(self.cx + X * self.F / Z, self.cy - Y * self.F / Z) for X, Y, Z in cv]
        if is_big(pts, self.W, self.H):
            pts = clip_rect(pts, self.W, self.H)
            if len(pts) < 3:
                return
        pygame.draw.polygon(self.surf, col, pts)

    # ---- objects --------------------------------------------------------
    def add_mesh(self, mesh, ox, oy, oz, f, u, r, flash=0.0):
        cam = self.cam
        px, py, pz = cam.pos
        cr, cu, cf = cam.r, cam.u, cam.f
        dx, dy, dz = ox - px, oy - py, oz - pz
        cz = dx * cf[0] + dy * cf[1] + dz * cf[2]
        rad = mesh.radius
        if cz < -rad or cz > self.far + rad:
            return
        cxx = dx * cr[0] + dy * cr[1] + dz * cr[2]
        if abs(cxx) > (cz + rad) * self.tx + rad:
            return
        cyy = dx * cu[0] + dy * cu[1] + dz * cu[2]
        if abs(cyy) > (cz + rad) * self.ty + rad:
            return
        lx = -(dx * r[0] + dy * r[1] + dz * r[2])
        ly = -(dx * u[0] + dy * u[1] + dz * u[2])
        lz = -(dx * f[0] + dy * f[1] + dz * f[2])
        Lx, Ly, Lz = dot(LIGHT, r), dot(LIGHT, u), dot(LIGHT, f)
        m00, m01, m02 = dot(cr, r), dot(cr, u), dot(cr, f)
        m10, m11, m12 = dot(cu, r), dot(cu, u), dot(cu, f)
        m20, m21, m22 = dot(cf, r), dot(cf, u), dot(cf, f)
        near, F, cx, cy = self.near, self.F, self.cx, self.cy
        cv, pj = [], []
        for (vx, vy, vz) in mesh.v:
            X = cxx + m00 * vx + m01 * vy + m02 * vz
            Y = cyy + m10 * vx + m11 * vy + m12 * vz
            Z = cz + m20 * vx + m21 * vy + m22 * vz
            cv.append((X, Y, Z))
            if Z >= near:
                k = F / Z
                pj.append((cx + X * k, cy - Y * k))
            else:
                pj.append(None)
        fs, fe = self.fog_s, self.fog_e
        inv = 1.0 / (fe - fs)
        items = self.items
        for (idx, col, nx, ny, nz, d, two) in mesh.faces:
            s = nx * lx + ny * ly + nz * lz - d
            lam = nx * Lx + ny * Ly + nz * Lz
            if s <= 0:
                if not two:
                    continue
                lam = -lam
            ok = True
            zs = 0.0
            for i in idx:
                if pj[i] is None:
                    ok = False
                zs += cv[i][2]
            zavg = zs / len(idx)
            if zavg > fe:
                continue
            if ok:
                pts = [pj[i] for i in idx]
            else:
                poly = clip_near([cv[i] for i in idx], near)
                if len(poly) < 3:
                    continue
                pts = [(cx + X * F / Z, cy - Y * F / Z) for X, Y, Z in poly]
            if is_big(pts, self.W, self.H):
                pts = clip_rect(pts, self.W, self.H)
                if len(pts) < 3:
                    continue
            sh = 0.5 + 0.75 * (lam if lam > 0 else 0.0) + flash
            c = (min(255, int(col[0] * sh)), min(255, int(col[1] * sh)), min(255, int(col[2] * sh)))
            if zavg > fs:
                c = mix(c, HAZE, (zavg - fs) * inv)
            items.append((zavg, 0, c, pts, 0))

    def to_cam(self, x, y, z):
        cam = self.cam
        dx, dy, dz = x - cam.pos[0], y - cam.pos[1], z - cam.pos[2]
        return (dx * cam.r[0] + dy * cam.r[1] + dz * cam.r[2],
                dx * cam.u[0] + dy * cam.u[1] + dz * cam.u[2],
                dx * cam.f[0] + dy * cam.f[1] + dz * cam.f[2])

    def add_circle(self, x, y, z, rad, color, fog=True, minz=24.0):
        X, Y, Z = self.to_cam(x, y, z)
        if Z < minz or Z > self.far:
            return
        k = self.F / Z
        sx, sy, rr = self.cx + X * k, self.cy - Y * k, rad * k
        if sx < -rr or sx > self.W + rr or sy < -rr or sy > self.H + rr or rr < 0.6:
            return
        if rr > 1400:
            rr = 1400
        if fog and Z > self.fog_s:
            color = mix(color, HAZE, clamp((Z - self.fog_s) / (self.fog_e - self.fog_s), 0, 1))
        self.items.append((Z, 1, color, (int(sx), int(sy)), max(1, int(rr))))

    def add_line(self, p0, p1, color, width):
        a = self.to_cam(*p0)
        b = self.to_cam(*p1)
        near = self.near
        if a[2] < near and b[2] < near:
            return
        if a[2] < near:
            t = (near - a[2]) / (b[2] - a[2])
            a = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, near)
        elif b[2] < near:
            t = (near - b[2]) / (a[2] - b[2])
            b = (b[0] + (a[0] - b[0]) * t, b[1] + (a[1] - b[1]) * t, near)
        F, cx, cy = self.F, self.cx, self.cy
        pa = (cx + a[0] * F / a[2], cy - a[1] * F / a[2])
        pb = (cx + b[0] * F / b[2], cy - b[1] * F / b[2])
        self.items.append((min(a[2], b[2]), 2, color, (pa, pb), width))

    def project(self, x, y, z):
        X, Y, Z = self.to_cam(x, y, z)
        if Z < self.near:
            return None, X, Y, Z
        k = self.F / Z
        return (self.cx + X * k, self.cy - Y * k), X, Y, Z

    def flush(self):
        items = self.items
        items.sort(key=itemgetter(0), reverse=True)
        surf = self.surf
        poly, circ, line = pygame.draw.polygon, pygame.draw.circle, pygame.draw.line
        for it in items:
            k = it[1]
            if k == 0:
                poly(surf, it[2], it[3])
            elif k == 1:
                circ(surf, it[2], it[3], it[4])
            else:
                line(surf, it[2], it[3][0], it[3][1], it[4])
        self.items = []


# --------------------------------------------------------------------------
# game entities
# --------------------------------------------------------------------------
class Body:
    def __init__(self, p, f=(0.0, 0.0, 1.0)):
        self.p = list(p)
        f, u, r = ortho(f, (0.0, 1.0, 0.0))
        self.f, self.u, self.r = f, u, r
        self.bank = 0.0
        self.flash = 0.0
        self.speed = 330.0


class Player(Body):
    def __init__(self, p):
        Body.__init__(self, p)
        self.hp = 100.0
        self.missiles = 6
        self.boost = 100.0
        self.gun_cd = 0.0
        self.msl_cd = 0.0
        self.msl_regen = 0.0
        self.alive = True


class Enemy(Body):
    def __init__(self, p, f, wave):
        Body.__init__(self, p, f)
        w = min(wave, 8)
        self.hp = 22.0 + 3.0 * w
        self.speed = 300.0 + 11.0 * w
        self.rate = 1.35 + 0.07 * w
        self.cool = random.uniform(1.0, 3.0)
        self.burst = 0
        self.aim_err = max(0.012, 0.055 - 0.004 * w)
        self.alive = True
        self.smoke = 0.0
        self.break_t = 0.0


class Missile:
    __slots__ = ("p", "f", "u", "r", "speed", "target", "life", "smoke", "bank")

    def __init__(self, p, f, target, speed):
        self.p = list(p)
        self.f, self.u, self.r = ortho(f, (0.0, 1.0, 0.0))
        self.speed = speed
        self.target = target
        self.life = 6.0
        self.smoke = 0.0
        self.bank = 0.0


class Bullet:
    __slots__ = ("p", "v", "life", "mine")

    def __init__(self, p, v, life, mine):
        self.p = list(p)
        self.v = v
        self.life = life
        self.mine = mine


# --------------------------------------------------------------------------
# game
# --------------------------------------------------------------------------
class Game:
    def __init__(self, args):
        self.args = args
        pygame.display.init()
        pygame.font.init()
        q = "high" if args.hd else args.quality
        self.quality = q
        RW, RH = {"low": (960, 540), "med": (1280, 720), "high": (1920, 1080)}[q]
        if args.res:
            RW, RH = [int(v) for v in args.res.lower().split("x")]
        self.RW, self.RH = RW, RH
        self.sc = RH / 720.0
        flags = pygame.SCALED | (pygame.FULLSCREEN if args.fullscreen else 0)
        try:
            self.screen = pygame.display.set_mode((RW, RH), flags)
        except pygame.error:
            self.screen = pygame.display.set_mode((RW, RH))
        pygame.display.set_caption("Sky Strike 3D")
        self.clock = pygame.time.Clock()
        sc = self.sc
        self.f_s = pygame.font.Font(None, int(26 * sc))
        self.f_m = pygame.font.Font(None, int(38 * sc))
        self.f_l = pygame.font.Font(None, int(74 * sc))
        self.f_xl = pygame.font.Font(None, int(150 * sc))

        random.seed(args.seed)
        self.world = World(seed=3)
        self.scene = Scene3D(self.screen, self.world, q)
        self.jet_player = make_jet((205, 210, 220), (60, 100, 205), (225, 60, 50))
        self.jet_enemy = make_jet((92, 96, 104), (175, 40, 40), (240, 150, 30))
        self.missile_mesh = make_missile()
        self.clouds = []
        crng = random.Random(11)
        for _ in range(26):
            cx, cy, cz = crng.uniform(0, TW), crng.uniform(1300, 2500), crng.uniform(0, TW)
            puffs = [(crng.uniform(-260, 260), crng.uniform(-40, 50), crng.uniform(-260, 260),
                      crng.uniform(90, 190)) for _ in range(crng.randint(6, 10))]
            self.clouds.append((cx, cy, cz, puffs))
        self.make_sounds()
        # input
        self.pointers = {}
        self.stick_id = None
        self.stick_o = (0, 0)
        self.touch_mode = False
        self.auto_fire = False
        self.invert = False
        self.show_fps = args.fps
        self.view = 0
        self.build_ui()
        self.best = self.load_best()
        self.state = "menu"
        self.t = 0.0
        self.msg = ("", 0.0)
        self.demo = args.demo
        self.cam_f, self.cam_u = (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)
        self.reset_game()
        if self.demo:
            self.state = "play"
        self.fps_acc = 0.0

    # ---- sound ----------------------------------------------------------
    def make_sounds(self):
        self.snd = {}
        if self.args.mute:
            return
        try:
            pygame.mixer.pre_init(22050, -16, 1, 512)
            pygame.mixer.init()
            fr, _, ch = pygame.mixer.get_init()
            pygame.mixer.set_num_channels(16)
        except Exception:
            return

        def mk(fn, dur, vol=1.0):
            n = int(fr * dur)
            data = array.array("h")
            for i in range(n):
                s = clamp(fn(i / fr, i / n), -1.0, 1.0) * vol
                v = int(s * 30000)
                data.append(v)
                if ch == 2:
                    data.append(v)
            return pygame.mixer.Sound(buffer=data.tobytes())

        rnd = random.uniform
        lp = [0.0]

        def gun(t, p):
            return (rnd(-1, 1) * 0.6 + math.sin(2 * math.pi * 170 * t) * 0.4) * math.exp(-t * 55)

        def missile(t, p):
            lp[0] += (rnd(-1, 1) - lp[0]) * (0.25 + 0.4 * p)
            return lp[0] * (1 - p) * 1.3 + math.sin(2 * math.pi * (1000 - 700 * p) * t) * 0.1 * (1 - p)

        def boom(t, p):
            lp[0] += (rnd(-1, 1) - lp[0]) * 0.07
            return lp[0] * math.exp(-t * 3.0) * 4.0

        def hit(t, p):
            return math.sin(2 * math.pi * 620 * t) * math.exp(-t * 40) * 0.8

        def warn(t, p):
            return math.sin(2 * math.pi * 880 * t) * (1 if (t * 9) % 1 < 0.6 else 0) * 0.5

        try:
            self.snd = {"gun": mk(gun, 0.08, 0.35), "msl": mk(missile, 0.7, 0.5),
                        "boom": mk(boom, 1.1, 0.7), "hit": mk(hit, 0.08, 0.4),
                        "warn": mk(warn, 0.25, 0.35)}
        except Exception:
            self.snd = {}

    def play(self, name):
        s = self.snd.get(name)
        if s:
            try:
                s.play()
            except Exception:
                pass

    def load_best(self):
        try:
            with open(os.path.expanduser("~/.skystrike_best")) as f:
                return int(f.read().strip())
        except Exception:
            return 0

    def save_best(self):
        try:
            with open(os.path.expanduser("~/.skystrike_best"), "w") as f:
                f.write(str(self.best))
        except Exception:
            pass

    # ---- ui -------------------------------------------------------------
    def build_ui(self):
        sc, RW, RH = self.sc, self.RW, self.RH
        self.btn = {
            "fire": (RW - 160 * sc, RH - 150 * sc, 100 * sc, "FIRE"),
            "msl": (RW - 385 * sc, RH - 95 * sc, 68 * sc, "MSL"),
            "boost": (RW - 140 * sc, RH - 350 * sc, 62 * sc, "BOOST"),
            "brake": (RW - 330 * sc, RH - 245 * sc, 54 * sc, "BRAKE"),
            "pause": (56 * sc, 175 * sc, 34 * sc, "II"),
            "cam": (146 * sc, 175 * sc, 34 * sc, "CAM"),
            "auto": (236 * sc, 175 * sc, 34 * sc, "AUTO"),
        }
        self.stick_r = 120 * sc

    def hit_button(self, pos):
        for name, (x, y, r, _) in self.btn.items():
            rr = r + 18 * self.sc
            if (pos[0] - x) ** 2 + (pos[1] - y) ** 2 <= rr * rr:
                return name
        return None

    def held(self, name):
        x, y, r, _ = self.btn[name]
        rr = r + 18 * self.sc
        for pid, pos in self.pointers.items():
            if pid == self.stick_id:
                continue
            if (pos[0] - x) ** 2 + (pos[1] - y) ** 2 <= rr * rr:
                return True
        return False

    def pointer_down(self, pid, pos):
        self.touch_mode = self.touch_mode or pid != "m"
        if self.state == "menu":
            self.start_game()
            return
        if self.state == "dead" and self.dead_t > 2.0:
            self.start_game()
            return
        b = self.hit_button(pos)
        if b == "msl":
            self.fire_missile()
        elif b == "pause":
            self.toggle_pause()
        elif b == "cam":
            self.view = (self.view + 1) % 3
        elif b == "auto":
            self.auto_fire = not self.auto_fire
        elif b is None and pos[0] < self.RW * 0.55 and self.stick_id is None:
            self.stick_id = pid
            self.stick_o = pos

    def toggle_pause(self):
        if self.state == "play":
            self.state = "pause"
        elif self.state == "pause":
            self.state = "play"

    def handle_events(self):
        RW, RH = self.RW, self.RH
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN:
                k = e.key
                if k == pygame.K_ESCAPE:
                    return False
                if k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if self.state == "menu" or (self.state == "dead" and self.dead_t > 1.5):
                        self.start_game()
                elif k == pygame.K_p:
                    self.toggle_pause()
                elif k in (pygame.K_m, pygame.K_k):
                    self.fire_missile()
                elif k == pygame.K_c:
                    self.view = (self.view + 1) % 3
                elif k == pygame.K_t:
                    self.auto_fire = not self.auto_fire
                elif k == pygame.K_i:
                    self.invert = not self.invert
                elif k == pygame.K_f:
                    pygame.display.toggle_fullscreen()
                elif k == pygame.K_F3:
                    self.show_fps = not self.show_fps
            elif e.type == pygame.FINGERDOWN:
                p = (e.x * RW, e.y * RH)
                pid = ("f", e.finger_id)
                self.pointers[pid] = p
                self.pointer_down(pid, p)
            elif e.type == pygame.FINGERMOTION:
                self.pointers[("f", e.finger_id)] = (e.x * RW, e.y * RH)
            elif e.type == pygame.FINGERUP:
                pid = ("f", e.finger_id)
                self.pointers.pop(pid, None)
                if self.stick_id == pid:
                    self.stick_id = None
            elif e.type == pygame.MOUSEBUTTONDOWN and not getattr(e, "touch", False):
                if e.button == 1:
                    self.pointers["m"] = e.pos
                    self.pointer_down("m", e.pos)
            elif e.type == pygame.MOUSEMOTION and not getattr(e, "touch", False):
                if "m" in self.pointers:
                    self.pointers["m"] = e.pos
            elif e.type == pygame.MOUSEBUTTONUP and not getattr(e, "touch", False):
                self.pointers.pop("m", None)
                if self.stick_id == "m":
                    self.stick_id = None
        return True

    def read_input(self):
        k = pygame.key.get_pressed()
        roll = (1 if (k[pygame.K_d] or k[pygame.K_RIGHT]) else 0) - (1 if (k[pygame.K_a] or k[pygame.K_LEFT]) else 0)
        pitch = (1 if (k[pygame.K_w] or k[pygame.K_UP]) else 0) - (1 if (k[pygame.K_s] or k[pygame.K_DOWN]) else 0)
        yaw = (1 if k[pygame.K_e] else 0) - (1 if k[pygame.K_q] else 0)
        boost = k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT]
        brake = k[pygame.K_LCTRL] or k[pygame.K_x]
        fire = k[pygame.K_SPACE] or k[pygame.K_j]
        roll, pitch = float(roll), float(pitch)
        if self.invert:
            pitch = -pitch
        if self.stick_id in self.pointers:
            p = self.pointers[self.stick_id]
            dx = clamp((p[0] - self.stick_o[0]) / self.stick_r, -1.0, 1.0)
            dy = clamp((p[1] - self.stick_o[1]) / self.stick_r, -1.0, 1.0)
            if abs(dx) > 0.12:
                roll = dx
            if abs(dy) > 0.12:
                pitch = -dy * (-1 if self.invert else 1)
        if self.touch_mode:
            boost = boost or self.held("boost")
            brake = brake or self.held("brake")
            fire = fire or self.held("fire")
        return roll, pitch, float(yaw), bool(boost), bool(brake), bool(fire)

    # ---- game flow ------------------------------------------------------
    def start_game(self):
        self.reset_game()
        self.state = "play"
        self.say("WAVE 1", 2.5)

    def say(self, text, t=2.0):
        self.msg = (text, t)

    def reset_game(self):
        w = self.world
        x, z, yaw = self.find_spawn()
        y = w.ground_h(x, z) + 650.0
        self.player = Player((x, y, z))
        self.player.f, self.player.u, self.player.r = ortho((math.sin(yaw), 0.0, math.cos(yaw)), (0.0, 1.0, 0.0))
        self.enemies = []
        self.bullets = []
        self.missiles = []
        self.particles = []
        self.score = 0
        self.wave = 0
        self.wave_timer = 0.5
        self.lock_target = None
        self.lock_t = 0.0
        self.shake = 0.0
        self.hit_flash = 0.0
        self.dead_t = 0.0
        self.warn_t = 0.0
        self.kills = 0
        self.cam_f, self.cam_u = self.player.f, self.player.u
        self.scene.cam.pos = tuple(self.player.p)

    def find_spawn(self):
        """a start position with open air straight ahead (no mountain in the way)"""
        rng = random.Random(5)
        w = self.world
        best = None
        for _ in range(120):
            x, z, yaw = rng.uniform(0, TW), rng.uniform(0, TW), rng.uniform(0, 2 * math.pi)
            y0 = w.ground_h(x, z) + 650.0
            worst = -1e9
            for k in range(0, 17):
                d = k * 250.0
                worst = max(worst, w.ground_h(x + math.sin(yaw) * d, z + math.cos(yaw) * d) - y0)
            if worst < -250 and w.terrain(x, z) > WATER + 30:
                return x, z, yaw
            if best is None or worst < best[0]:
                best = (worst, x, z, yaw)
        return best[1], best[2], best[3]

    def spawn_wave(self):
        self.wave += 1
        n = min(2 + self.wave, 8)
        p = self.player
        for i in range(n):
            a = random.uniform(0, 2 * math.pi)
            d = random.uniform(3000, 4200)
            x, z = p.p[0] + math.cos(a) * d, p.p[2] + math.sin(a) * d
            y = max(p.p[1] + random.uniform(-300, 400), self.world.ground_h(x, z) + 600)
            f = norm((p.p[0] - x, 0.0, p.p[2] - z))
            self.enemies.append(Enemy((x, y, z), f, self.wave))
        self.say("WAVE %d  -  %d ENEMIES" % (self.wave, n), 2.8)

    # ---- effects --------------------------------------------------------
    def particle(self, x, y, z, vx, vy, vz, life, size, kind):
        if len(self.particles) < 450:
            self.particles.append([x, y, z, vx, vy, vz, 0.0, life, size, kind])

    def explode(self, p, big=1.0):
        R = random.uniform
        for _ in range(int(26 * big)):
            self.particle(p[0], p[1], p[2], R(-160, 160) * big, R(-120, 160) * big, R(-160, 160) * big,
                          R(0.5, 1.2), R(25, 60) * big, 0)
        for _ in range(int(14 * big)):
            self.particle(p[0], p[1], p[2], R(-90, 90), R(10, 120), R(-90, 90), R(1.4, 2.8), R(40, 80) * big, 1)
        for _ in range(int(16 * big)):
            self.particle(p[0], p[1], p[2], R(-420, 420), R(-300, 420), R(-420, 420), R(0.3, 0.7), R(6, 10), 2)
        self.play("boom")
        d = math.dist(p, self.player.p)
        self.shake = max(self.shake, clamp(1.4 - d / 2500.0, 0.0, 1.0) * big)

    def spark(self, p, n=4):
        R = random.uniform
        for _ in range(n):
            self.particle(p[0], p[1], p[2], R(-200, 200), R(-200, 200), R(-200, 200), R(0.15, 0.35), R(4, 7), 2)

    # ---- combat ---------------------------------------------------------
    def fire_missile(self):
        p = self.player
        if self.state != "play" or not p.alive or p.missiles <= 0 or p.msl_cd > 0:
            return
        p.missiles -= 1
        p.msl_cd = 0.7
        tgt = self.lock_target if (self.lock_target and self.lock_t >= 0.7 and self.lock_target.alive) else None
        pos = lc(tuple(p.p), 1.0, p.u, -9.0)
        self.missiles.append(Missile(pos, p.f, tgt, p.speed + 120))
        self.play("msl")

    def fire_gun(self):
        p = self.player
        if p.gun_cd > 0 or not p.alive:
            return
        p.gun_cd = 0.075
        for s in (-1, 1):
            pos = lc(lc(tuple(p.p), 1.0, p.r, s * 11.0), 1.0, p.f, 18.0)
            self.bullets.append(Bullet(pos, lc(p.f, 1900.0 + p.speed, (0, 0, 0), 0), 1.2, True))
        self.gun_sound = getattr(self, "gun_sound", 0) + 1
        if self.gun_sound % 3 == 0:
            self.play("gun")

    def damage_player(self, dmg):
        p = self.player
        if not p.alive:
            return
        p.hp -= dmg
        p.flash = 0.25
        self.hit_flash = 0.35
        self.shake = max(self.shake, 0.35)
        self.play("hit")
        if p.hp <= 0:
            self.kill_player()

    def kill_player(self):
        p = self.player
        if not p.alive:
            return
        p.alive = False
        p.hp = 0
        self.view = 0
        self.explode(p.p, 2.2)
        self.state = "dead"
        self.dead_t = 0.0
        if self.score > self.best:
            self.best = self.score
            self.save_best()

    def kill_enemy(self, e):
        e.alive = False
        self.explode(e.p, 1.6)
        self.kills += 1
        self.score += 100 + 25 * self.wave
        self.say("TARGET DESTROYED  +%d" % (100 + 25 * self.wave), 1.4)

    # ---- update ---------------------------------------------------------
    def update_player(self, dt, inp):
        p = self.player
        roll, pitch, yaw, boost, brake, fire = inp
        if abs(roll) > 0.02:
            bank_s = -p.r[1]
            if not ((roll > 0 and bank_s > 0.92) or (roll < 0 and bank_s < -0.92)):
                b_roll(p, roll * 2.5 * dt)
        else:
            b_roll(p, p.r[1] * 1.8 * dt)
        if abs(pitch) > 0.02:
            b_pitch(p, pitch * 1.4 * dt)
        if abs(yaw) > 0.02:
            b_yaw(p, yaw * 0.7 * dt)
        if p.u[1] > 0.05:
            ang = -p.r[1] * 1.1 * dt
            p.f, p.u, p.r = rot_y(p.f, ang), rot_y(p.u, ang), rot_y(p.r, ang)
        if p.p[1] > 2600:
            b_pitch(p, -0.7 * dt)
        target = 330.0
        if boost and p.boost > 0:
            target = 570.0
            p.boost -= 38 * dt
            self.particle(p.p[0] - p.f[0] * 30, p.p[1] - p.f[1] * 30, p.p[2] - p.f[2] * 30,
                          0, 0, 0, 0.25, 9, 0)
        else:
            if brake:
                target = 190.0
            p.boost = min(100.0, p.boost + 14 * dt)
        p.speed += (target - p.speed) * min(1.0, 1.4 * dt)
        p.speed = clamp(p.speed - p.f[1] * 110 * dt, 150.0, 650.0)
        p.f, p.u, p.r = ortho(p.f, p.u)
        for i in range(3):
            p.p[i] += p.f[i] * p.speed * dt
        p.gun_cd = max(0.0, p.gun_cd - dt)
        p.msl_cd = max(0.0, p.msl_cd - dt)
        p.flash = max(0.0, p.flash - dt)
        p.msl_regen += dt
        if p.msl_regen > 9.0:
            p.msl_regen = 0.0
            if p.missiles < 8:
                p.missiles += 1
        if fire or (self.auto_fire and self.lock_target and self.lock_t > 0.15):
            if fire or math.dist(self.lock_target.p, p.p) < 2300:
                self.fire_gun()

    def update_demo_pilot(self, dt):
        p = self.player
        tgt = None
        best = 1e18
        for e in self.enemies:
            d = math.dist(e.p, p.p)
            if d < best:
                best, tgt = d, e
        if tgt:
            lead = min(best / 2200.0, 1.2)
            aim = [tgt.p[i] + tgt.f[i] * tgt.speed * lead for i in range(3)]
            d = norm((aim[0] - p.p[0], aim[1] - p.p[1], aim[2] - p.p[2]))
        else:
            d = norm((p.f[0], 0.0, p.f[2]))
        gh = self.world.ground_h(p.p[0] + p.f[0] * 500, p.p[2] + p.f[2] * 500)
        if p.p[1] < gh + 350:
            d = norm((d[0], 0.9, d[2]))
        elif p.p[1] > gh + 1500 and d[1] > -0.2:
            d = norm((d[0], -0.25, d[2]))
        steer(p, d, 2.0, dt)
        p.speed = 380.0
        for i in range(3):
            p.p[i] += p.f[i] * p.speed * dt
        p.gun_cd = max(0.0, p.gun_cd - dt)
        p.msl_cd = max(0.0, p.msl_cd - dt)
        if tgt and dot(p.f, d) > 0.985 and best < 2200:
            self.fire_gun()
        if tgt and self.lock_t > 0.8 and p.missiles > 0 and best < 3000:
            self.fire_missile()
        p.flash = max(0.0, p.flash - dt)

    def update_enemy(self, e, dt):
        p = self.player
        w = self.world
        to = (p.p[0] - e.p[0], p.p[1] - e.p[1], p.p[2] - e.p[2])
        dist = math.sqrt(dot(to, to)) + 1e-6
        lead = clamp(dist / 2000.0, 0.0, 1.0) * 0.6
        aim = (p.p[0] + p.f[0] * p.speed * lead, p.p[1] + p.f[1] * p.speed * lead, p.p[2] + p.f[2] * p.speed * lead)
        d = norm((aim[0] - e.p[0], aim[1] - e.p[1], aim[2] - e.p[2]))
        if not p.alive:
            d = norm((e.f[0], 0.15, e.f[2]))
        e.break_t -= dt
        if dist < 480 and dot(e.f, to) / dist > 0.3 and e.break_t <= 0:
            e.break_t = 1.6
        if e.break_t > 0:
            d = norm((-to[0] / dist + e.r[0] * 0.6, 0.45, -to[2] / dist + e.r[2] * 0.6))
        gh = w.ground_h(e.p[0] + e.f[0] * 350, e.p[2] + e.f[2] * 350)
        gh = max(gh, w.ground_h(e.p[0], e.p[2]))
        if e.p[1] < gh + 260:
            d = norm((e.f[0], 0.9, e.f[2]))
        steer(e, d, e.rate, dt)
        e.speed += (320 + 11 * min(self.wave, 8) - e.speed) * min(1.0, dt)
        for i in range(3):
            e.p[i] += e.f[i] * e.speed * dt
        e.flash = max(0.0, e.flash - dt)
        # shooting
        e.cool -= dt
        ang = dot(e.f, to) / dist
        if p.alive and e.burst > 0:
            e.cool = 0.0
        if p.alive and e.cool <= 0 and dist < 1900 and ang > 0.985 and e.break_t <= 0:
            if e.burst == 0:
                e.burst = 6
            e.burst -= 1
            e.cool = 0.1 if e.burst > 0 else random.uniform(1.4, 2.6)
            spread = (random.uniform(-e.aim_err, e.aim_err), random.uniform(-e.aim_err, e.aim_err),
                      random.uniform(-e.aim_err, e.aim_err))
            dirv = norm(lc(e.f, 1.0, spread, 1.0))
            v = lc(dirv, 1500.0 + e.speed, (0, 0, 0), 0)
            self.bullets.append(Bullet(lc(tuple(e.p), 1.0, e.f, 20.0), v, 1.5, False))
        if e.hp < 10 and random.random() < 0.6:
            self.particle(e.p[0], e.p[1], e.p[2], 0, 10, 0, 1.0, 18, 1)

    def update_bullets(self, dt):
        p = self.player
        keep = []
        for b in self.bullets:
            old = (b.p[0], b.p[1], b.p[2])
            b.p[0] += b.v[0] * dt
            b.p[1] += b.v[1] * dt
            b.p[2] += b.v[2] * dt
            b.life -= dt
            if b.life <= 0:
                continue
            hit = False
            if b.mine:
                for e in self.enemies:
                    if e.alive and seg_hit(old, b.p, e.p, 34.0):
                        e.hp -= 2.0
                        e.flash = 0.2
                        self.spark(b.p)
                        if e.hp <= 0:
                            self.kill_enemy(e)
                        hit = True
                        break
            elif p.alive and seg_hit(old, b.p, p.p, 26.0):
                self.damage_player(4.5)
                self.spark(b.p)
                hit = True
            if not hit:
                keep.append(b)
        self.bullets = keep

    def update_missiles(self, dt):
        keep = []
        for m in self.missiles:
            m.life -= dt
            m.speed = min(1250.0, m.speed + 700 * dt)
            t = m.target
            if t is not None and t.alive:
                to = (t.p[0] - m.p[0], t.p[1] - m.p[1], t.p[2] - m.p[2])
                d = norm(to)
                steer(m, d, 2.6, dt, 0.3)
            for i in range(3):
                m.p[i] += m.f[i] * m.speed * dt
            m.smoke -= dt
            if m.smoke <= 0:
                m.smoke = 0.025
                self.particle(m.p[0] - m.f[0] * 14, m.p[1] - m.f[1] * 14, m.p[2] - m.f[2] * 14,
                              random.uniform(-8, 8), random.uniform(-8, 8), random.uniform(-8, 8), 1.3, 7, 1)
            boom = False
            for e in self.enemies:
                if e.alive and math.dist(e.p, m.p) < 55:
                    e.hp -= 40
                    e.flash = 0.3
                    boom = True
                    if e.hp <= 0:
                        self.kill_enemy(e)
                    break
            if not boom and m.p[1] < self.world.ground_h(m.p[0], m.p[2]) + 2:
                boom = True
            if boom:
                self.explode(m.p, 0.8)
            elif m.life > 0:
                keep.append(m)
        self.missiles = keep

    def update_lock(self, dt):
        p = self.player
        best, bc = None, 0.935
        for e in self.enemies:
            if not e.alive:
                continue
            v = (e.p[0] - p.p[0], e.p[1] - p.p[1], e.p[2] - p.p[2])
            d = math.sqrt(dot(v, v)) + 1e-6
            if 120 < d < 3800:
                c = dot(p.f, v) / d
                if c > bc:
                    bc, best = c, e
        if best is self.lock_target and best is not None:
            self.lock_t += dt
        else:
            self.lock_target = best
            self.lock_t = 0.0

    def update_particles(self, dt):
        keep = []
        for q in self.particles:
            q[6] += dt
            if q[6] >= q[7]:
                continue
            q[0] += q[3] * dt
            q[1] += q[4] * dt
            q[2] += q[5] * dt
            if q[9] == 1:
                q[4] += 20 * dt
            keep.append(q)
        self.particles = keep

    def update(self, dt):
        self.t += dt
        p = self.player
        mt = self.msg[1]
        if mt > 0:
            self.msg = (self.msg[0], mt - dt)
        self.hit_flash = max(0.0, self.hit_flash - dt)
        self.shake = max(0.0, self.shake - dt * 1.6)
        if self.state in ("play", "dead", "menu"):
            if self.state == "menu":
                steer(p, norm((math.sin(self.t * 0.25), 0.02 + 0.05 * math.sin(self.t * 0.5), math.cos(self.t * 0.25))), 0.6, dt)
                p.speed = 260.0
                for i in range(3):
                    p.p[i] += p.f[i] * p.speed * dt
            elif self.state == "play" and p.alive:
                if self.demo:
                    self.update_demo_pilot(dt)
                else:
                    self.update_player(dt, self.read_input())
                gh = self.world.ground_h(p.p[0], p.p[2])
                if p.p[1] < gh + 10:
                    self.kill_player()
                    self.say("CRASHED", 3)
                elif p.p[1] < gh + 230 and p.f[1] < -0.12:
                    self.warn_t -= dt
                    if self.warn_t <= 0:
                        self.warn_t = 0.45
                        self.play("warn")
            if self.state == "dead":
                self.dead_t += dt
            if self.state != "menu":
                self.update_lock(dt)
                if not self.enemies and self.state == "play":
                    self.wave_timer -= dt
                    if self.wave_timer <= 0:
                        if self.wave > 0:
                            p.hp = min(100.0, p.hp + 25)
                            p.missiles = min(8, p.missiles + 2)
                            self.score += 250
                        self.spawn_wave()
                        self.wave_timer = 3.0
                for e in self.enemies:
                    self.update_enemy(e, dt)
                    if e.alive and p.alive and math.dist(e.p, p.p) < 48:
                        e.alive = False
                        self.explode(e.p, 1.5)
                        self.damage_player(45)
                    elif e.alive and e.p[1] < self.world.ground_h(e.p[0], e.p[2]) + 8:
                        e.alive = False
                        self.explode(e.p, 1.2)
                        self.score += 50
                self.enemies = [e for e in self.enemies if e.alive]
                if not self.enemies:
                    self.wave_timer = min(self.wave_timer, 3.0)
                self.update_bullets(dt)
                self.update_missiles(dt)
        self.update_particles(dt)
        self.update_camera(dt)

    def update_camera(self, dt):
        p = self.player
        cam = self.scene.cam
        if self.view == 1 and p.alive:
            f, u, r = p.f, p.u, p.r
            pos = lc(lc(tuple(p.p), 1.0, f, 7.0), 1.0, u, 3.2)
        else:
            k = min(1.0, 7.0 * dt)
            tf, tu = p.f, p.u
            if not p.alive:
                tf, tu = self.cam_f, self.cam_u
            self.cam_f, self.cam_u, _ = ortho(lc(self.cam_f, 1 - k, tf, k), lc(self.cam_u, 1 - k, tu, k))
            f, u, r = ortho(self.cam_f, self.cam_u)
            dist, h = (88.0, 16.0) if self.view == 0 else (190.0, 42.0)
            if p.alive:
                self.cam_anchor = tuple(p.p)
            pos = lc(lc(self.cam_anchor if hasattr(self, "cam_anchor") else tuple(p.p), 1.0, f, -dist), 1.0, u, h)
            ca, sa = math.cos(0.11), math.sin(0.11)      # look slightly down: more ground on screen
            f, u = lc(f, ca, u, -sa), lc(u, ca, f, sa)
        if self.shake > 0:
            s = self.shake * 9.0
            pos = (pos[0] + random.uniform(-s, s), pos[1] + random.uniform(-s, s), pos[2] + random.uniform(-s, s))
        cam.pos, cam.f, cam.u, cam.r = pos, f, u, r

    # ---- rendering ------------------------------------------------------
    FIRE = [(255, 252, 215), (255, 232, 140), (255, 196, 70), (250, 140, 40), (230, 90, 30),
            (180, 60, 30), (110, 50, 40), (70, 60, 60)]
    SMOKE = [(95, 95, 100), (120, 122, 128), (150, 154, 160), (175, 182, 190), (190, 202, 214)]
    SPARK = [(255, 255, 220), (255, 230, 130), (255, 180, 60), (240, 110, 30)]

    def draw_scene(self):
        sc = self.scene
        cam = sc.cam
        sc.draw_sky()
        sc.draw_ground()
        p = self.player
        cp = cam.pos
        if p.alive and p.p[1] - self.world.ground_h(p.p[0], p.p[2]) < 1500:
            sc.draw_shadow(p.p[0], p.p[2], p.f[0], p.f[2])
        for e in self.enemies:
            if e.p[1] - self.world.ground_h(e.p[0], e.p[2]) < 1500 and math.dist(e.p, cp) < 3500:
                sc.draw_shadow(e.p[0], e.p[2], e.f[0], e.f[2])
        # static scenery
        ci, cj = int(math.floor(cp[0] / TILE)), int(math.floor(cp[2] / TILE))
        R = sc.R
        stt = self.world.statics
        for ti in range(ci - R, ci + R + 1):
            for tj in range(cj - R, cj + R + 1):
                lst = stt.get((ti % NT, tj % NT))
                if not lst:
                    continue
                dd = max(abs(ti - ci), abs(tj - cj))
                for (kind, ox, oy, oz, mesh) in lst:
                    if kind == "t" and dd > sc.tree_range:
                        continue
                    if kind == "c" and dd > sc.city_range:
                        continue
                    sc.add_mesh(mesh, ti * TILE + ox, oy, tj * TILE + oz, (0, 0, 1), (0, 1, 0), (1, 0, 0))
        # clouds
        for (cx, cy, cz, puffs) in self.clouds:
            dx = (cx - cp[0] + TW / 2) % TW - TW / 2
            dz = (cz - cp[2] + TW / 2) % TW - TW / 2
            if abs(dx) > sc.far or abs(dz) > sc.far:
                continue
            for (ox, oy, oz, rr) in puffs:
                top = oy > -10
                sc.add_circle(cp[0] + dx + ox, cy + oy, cp[2] + dz + oz, rr,
                              (247, 249, 252) if top else (206, 216, 230))
        # planes
        if p.alive and self.view != 1:
            sc.add_mesh(self.jet_player, p.p[0], p.p[1], p.p[2], p.f, p.u, p.r, 0.5 if p.flash > 0 else 0.0)
            tail = lc(tuple(p.p), 1.0, p.f, -22.0)
            sc.add_circle(tail[0], tail[1], tail[2], 1.2 + 1.2 * (p.speed > 450), (255, 190, 80), False)
        for e in self.enemies:
            sc.add_mesh(self.jet_enemy, e.p[0], e.p[1], e.p[2], e.f, e.u, e.r, 0.6 if e.flash > 0 else 0.0)
            tail = lc(tuple(e.p), 1.0, e.f, -22.0)
            sc.add_circle(tail[0], tail[1], tail[2], 1.3, (255, 170, 70), False)
        for m in self.missiles:
            sc.add_mesh(self.missile_mesh, m.p[0], m.p[1], m.p[2], m.f, m.u, m.r)
            tail = lc(tuple(m.p), 1.0, m.f, -17.0)
            sc.add_circle(tail[0], tail[1], tail[2], 2.5, (255, 210, 120), False)
        for b in self.bullets:
            tail = (b.p[0] - b.v[0] * 0.035, b.p[1] - b.v[1] * 0.035, b.p[2] - b.v[2] * 0.035)
            sc.add_line(tail, b.p, (255, 240, 120) if b.mine else (255, 80, 60), 3 if self.RH > 800 else 2)
        for q in self.particles:
            t = q[6] / q[7]
            kind = q[9]
            if kind == 0:
                col = self.FIRE[min(len(self.FIRE) - 1, int(t * len(self.FIRE)))]
                rad = q[8] * (1.0 - 0.45 * t)
            elif kind == 1:
                col = self.SMOKE[min(len(self.SMOKE) - 1, int(t * len(self.SMOKE)))]
                rad = q[8] * (0.5 + 0.9 * t)
            else:
                col = self.SPARK[min(len(self.SPARK) - 1, int(t * len(self.SPARK)))]
                rad = q[8] * (1 - 0.5 * t)
            sc.add_circle(q[0], q[1], q[2], rad, col, False, 90.0)
        sc.flush()

    def text(self, s, font, color, pos, anchor="topleft", shadow=True):
        img = font.render(s, True, color)
        r = img.get_rect(**{anchor: pos})
        if shadow:
            self.screen.blit(font.render(s, True, (0, 0, 0)), r.move(2, 2))
        self.screen.blit(img, r)

    def draw_hud(self):
        sc, RW, RH = self.sc, self.RW, self.RH
        scr = self.screen
        p = self.player
        sce = self.scene
        cam = sce.cam
        GREEN = (120, 255, 150)
        # damage flash vignette
        if self.hit_flash > 0:
            w = int(14 * sc * self.hit_flash / 0.35) + 2
            pygame.draw.rect(scr, (220, 30, 30), (0, 0, RW, RH), w)
        if self.view == 1:
            pygame.draw.polygon(scr, (22, 24, 28), [(0, RH), (0, RH - 120 * sc), (RW * 0.25, RH - 70 * sc),
                                                    (RW * 0.75, RH - 70 * sc), (RW, RH - 120 * sc), (RW, RH)])
            pygame.draw.polygon(scr, (22, 24, 28), [(0, 0), (RW, 0), (RW, 36 * sc), (RW * 0.7, 18 * sc),
                                                    (RW * 0.3, 18 * sc), (0, 36 * sc)])
            pygame.draw.line(scr, (22, 24, 28), (0, 0), (RW * 0.12, RH), int(14 * sc))
            pygame.draw.line(scr, (22, 24, 28), (RW, 0), (RW * 0.88, RH), int(14 * sc))
        # gun pipper
        if p.alive:
            pt, _, _, _ = sce.project(p.p[0] + p.f[0] * 1100, p.p[1] + p.f[1] * 1100, p.p[2] + p.f[2] * 1100)
            if pt:
                x, y = int(pt[0]), int(pt[1])
                r = int(16 * sc)
                pygame.draw.circle(scr, GREEN, (x, y), r, 2)
                pygame.draw.line(scr, GREEN, (x - r - 10 * sc, y), (x - r + 4 * sc, y), 2)
                pygame.draw.line(scr, GREEN, (x + r - 4 * sc, y), (x + r + 10 * sc, y), 2)
                pygame.draw.line(scr, GREEN, (x, y - r - 10 * sc), (x, y - r + 4 * sc), 2)
        # enemy markers
        for e in self.enemies:
            pt, X, Y, Z = sce.project(e.p[0], e.p[1], e.p[2])
            dist = math.dist(e.p, p.p)
            locked = e is self.lock_target
            if pt and 0 <= pt[0] <= RW and 0 <= pt[1] <= RH:
                x, y = int(pt[0]), int(pt[1])
                s = int(max(14, min(46, 5200.0 / max(dist, 100.0))) * sc)
                col = (255, 60, 60) if (locked and self.lock_t >= 0.7) else ((255, 200, 60) if locked else (255, 110, 90))
                l = s // 3
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        cx_, cy_ = x + sx * s, y + sy * s
                        pygame.draw.line(scr, col, (cx_, cy_), (cx_ - sx * l, cy_), 2)
                        pygame.draw.line(scr, col, (cx_, cy_), (cx_, cy_ - sy * l), 2)
                self.text("%dm" % (dist * 0.5), self.f_s, col, (x, y + s + 4 * sc), "midtop", False)
                if locked:
                    self.text("LOCKED" if self.lock_t >= 0.7 else "LOCKING...", self.f_s, col,
                              (x, y - s - 6 * sc), "midbottom", False)
            else:
                ang = math.atan2(-Y if Z > 0 else Y, X if Z > 0 else -X) if (abs(X) + abs(Y)) > 1e-3 else 0
                ang = math.atan2(-Y, X) if Z > 0 else math.atan2(Y, -X)
                ex = RW / 2 + math.cos(ang) * (RW * 0.43)
                ey = RH / 2 + math.sin(ang) * (RH * 0.40)
                pts = [(ex + math.cos(ang) * 16 * sc, ey + math.sin(ang) * 16 * sc),
                       (ex + math.cos(ang + 2.5) * 14 * sc, ey + math.sin(ang + 2.5) * 14 * sc),
                       (ex + math.cos(ang - 2.5) * 14 * sc, ey + math.sin(ang - 2.5) * 14 * sc)]
                pygame.draw.polygon(scr, (255, 90, 70), pts)
        # radar
        rr = int(82 * sc)
        rx, ry = RW - rr - 24 * sc, rr + 24 * sc
        pygame.draw.circle(scr, (10, 40, 25), (rx, ry), rr)
        pygame.draw.circle(scr, (60, 170, 100), (rx, ry), rr, 2)
        pygame.draw.circle(scr, (40, 120, 70), (rx, ry), rr // 2, 1)
        pygame.draw.line(scr, (40, 120, 70), (rx - rr, ry), (rx + rr, ry), 1)
        pygame.draw.line(scr, (40, 120, 70), (rx, ry - rr), (rx, ry + rr), 1)
        fh = norm((p.f[0], 0, p.f[2]))
        rh = (fh[2], 0.0, -fh[0])
        for e in self.enemies:
            v = (e.p[0] - p.p[0], 0, e.p[2] - p.p[2])
            a, b = dot(v, fh), dot(v, rh)
            d = math.hypot(a, b)
            k = min(1.0, 5500.0 / max(d, 1.0)) * (1.0 / 5500.0)
            ex, ey = rx + b * k * rr, ry - a * k * rr
            if d > 5500:
                m = rr / d * 5500.0 / 5500.0
            pygame.draw.circle(scr, (255, 80, 60), (int(ex), int(ey)), max(3, int(4 * sc)))
        pygame.draw.polygon(scr, GREEN, [(rx, ry - 7 * sc), (rx - 5 * sc, ry + 5 * sc), (rx + 5 * sc, ry + 5 * sc)])
        # bars
        bx, by, bw, bh = 24 * sc, 24 * sc, 260 * sc, 22 * sc
        pygame.draw.rect(scr, (0, 0, 0), (bx - 2, by - 2, bw + 4, bh + 4), border_radius=5)
        hp = clamp(p.hp / 100.0, 0, 1)
        pygame.draw.rect(scr, (230, 60, 50) if hp < 0.3 else (90, 220, 110), (bx, by, bw * hp, bh), border_radius=4)
        self.text("HULL", self.f_s, (255, 255, 255), (bx + 8 * sc, by + 2 * sc), shadow=False)
        by2 = by + bh + 12 * sc
        pygame.draw.rect(scr, (0, 0, 0), (bx - 2, by2 - 2, bw + 4, 16 * sc + 4), border_radius=5)
        pygame.draw.rect(scr, (255, 170, 50), (bx, by2, bw * clamp(p.boost / 100.0, 0, 1), 16 * sc), border_radius=4)
        self.text("BOOST", self.f_s, (255, 255, 255), (bx + 8 * sc, by2 - 1), shadow=False)
        self.text("MISSILES", self.f_s, (255, 255, 255), (bx, by2 + 28 * sc))
        for i in range(8):
            c = (255, 230, 90) if i < p.missiles else (70, 70, 70)
            pygame.draw.polygon(scr, c, [(bx + 110 * sc + i * 18 * sc, by2 + 44 * sc),
                                         (bx + 114 * sc + i * 18 * sc, by2 + 28 * sc),
                                         (bx + 118 * sc + i * 18 * sc, by2 + 44 * sc)])
        # score / wave
        self.text("SCORE %d" % self.score, self.f_m, (255, 255, 255), (RW / 2, 18 * sc), "midtop")
        self.text("WAVE %d    KILLS %d    BEST %d" % (self.wave, self.kills, self.best), self.f_s,
                  (210, 220, 235), (RW / 2, 58 * sc), "midtop")
        # speed / alt
        alt = p.p[1] - self.world.ground_h(p.p[0], p.p[2])
        self.text("SPD %d km/h" % int(p.speed * 2.0), self.f_m, (140, 235, 255), (RW - 24 * sc, 24 * sc + 2 * rr + 12 * sc), "topright")
        self.text("ALT %d m" % int(alt * 0.5), self.f_m, (140, 235, 255), (RW - 24 * sc, 24 * sc + 2 * rr + 46 * sc), "topright")
        if alt < 260 and p.f[1] < -0.1 and p.alive:
            if int(self.t * 5) % 2 == 0:
                self.text("PULL UP!", self.f_l, (255, 60, 50), (RW / 2, RH * 0.28), "center")
        # message
        if self.msg[1] > 0:
            self.text(self.msg[0], self.f_l, (255, 235, 120), (RW / 2, RH * 0.2), "center")
        # touch ui
        if self.touch_mode or self.args.touch:
            held = set()
            for name in self.btn:
                if name in ("fire", "boost", "brake") and self.held(name):
                    held.add(name)
            for name, (x, y, r, label) in self.btn.items():
                on = name in held or (name == "auto" and self.auto_fire)
                s = pygame.Surface((int(r * 2), int(r * 2)), pygame.SRCALPHA)
                pygame.draw.circle(s, (255, 255, 255, 120 if on else 45), (int(r), int(r)), int(r))
                pygame.draw.circle(s, (255, 255, 255, 190), (int(r), int(r)), int(r), 3)
                scr.blit(s, (x - r, y - r))
                self.text(label, self.f_s if r < 70 * sc else self.f_m, (255, 255, 255), (x, y), "center", False)
            if self.stick_id in self.pointers:
                o = self.stick_o
                pos = self.pointers[self.stick_id]
                pygame.draw.circle(scr, (255, 255, 255), (int(o[0]), int(o[1])), int(self.stick_r), 3)
                dx, dy = pos[0] - o[0], pos[1] - o[1]
                l = math.hypot(dx, dy)
                if l > self.stick_r:
                    dx, dy = dx / l * self.stick_r, dy / l * self.stick_r
                pygame.draw.circle(scr, (255, 255, 255), (int(o[0] + dx), int(o[1] + dy)), int(40 * sc))
            else:
                self.text("drag here to fly", self.f_s, (255, 255, 255), (150 * sc, RH - 60 * sc), "midbottom")

    def draw_overlays(self):
        RW, RH, sc = self.RW, self.RH, self.sc
        dim = pygame.Surface((RW, RH), pygame.SRCALPHA)
        if self.state == "menu":
            dim.fill((0, 0, 0, 110))
            self.screen.blit(dim, (0, 0))
            self.text("SKY STRIKE 3D", self.f_xl, (255, 255, 255), (RW / 2, RH * 0.22), "center")
            self.text("Defeat endless waves of enemy fighters", self.f_m, (255, 220, 90), (RW / 2, RH * 0.36), "center")
            self.text("PRESS ENTER  /  TAP TO START", self.f_l, (140, 235, 255), (RW / 2, RH * 0.55), "center")
            lines = ["W/S pitch   A/D bank & turn   Q/E rudder   SHIFT boost   CTRL brake",
                     "SPACE machine gun   M homing missile (lock first)   C camera   T auto-fire",
                     "Touch: drag left half = stick, buttons on the right"]
            for i, l in enumerate(lines):
                self.text(l, self.f_s, (230, 230, 235), (RW / 2, RH * (0.70 + i * 0.05)), "center")
        elif self.state == "pause":
            dim.fill((0, 0, 0, 120))
            self.screen.blit(dim, (0, 0))
            self.text("PAUSED", self.f_xl, (255, 255, 255), (RW / 2, RH / 2), "center")
            self.text("press P to resume", self.f_m, (220, 220, 220), (RW / 2, RH * 0.62), "center")
        elif self.state == "dead" and self.dead_t > 1.2:
            dim.fill((0, 0, 0, min(150, int((self.dead_t - 1.2) * 120))))
            self.screen.blit(dim, (0, 0))
            self.text("GAME OVER", self.f_xl, (255, 80, 70), (RW / 2, RH * 0.3), "center")
            self.text("Score %d     Wave %d     Kills %d" % (self.score, self.wave, self.kills), self.f_l,
                      (255, 255, 255), (RW / 2, RH * 0.5), "center")
            self.text("Best %d" % self.best, self.f_m, (255, 220, 90), (RW / 2, RH * 0.6), "center")
            if self.dead_t > 2.0:
                self.text("PRESS ENTER  /  TAP TO FLY AGAIN", self.f_m, (140, 235, 255), (RW / 2, RH * 0.74), "center")

    def draw(self):
        self.draw_scene()
        if self.state != "menu":
            self.draw_hud()
        self.draw_overlays()
        if self.show_fps:
            self.text("%d FPS" % self.clock.get_fps(), self.f_s, (255, 255, 0), (10 * self.sc, self.RH - 30 * self.sc))
        pygame.display.flip()

    # ---- main loop ------------------------------------------------------
    def run(self):
        frames = 0
        while True:
            if not self.handle_events():
                break
            dt = self.clock.tick(self.args.maxfps) / 1000.0
            dt = min(dt, 0.05)
            if self.args.frames:
                dt = 1 / 30.0
            if self.state != "pause":
                self.update(dt)
            else:
                self.update_camera(0)
            self.draw()
            frames += 1
            if self.args.shot_at and frames == self.args.shot_at:
                pygame.image.save(self.screen, self.args.shot)
            if self.args.frames and frames >= self.args.frames:
                break
        if self.args.shot and not self.args.shot_at:
            pygame.image.save(self.screen, self.args.shot)


def main():
    ap = argparse.ArgumentParser(description="Sky Strike 3D - fighter jet combat")
    ap.add_argument("--quality", choices=["low", "med", "high"], default="med")
    ap.add_argument("--hd", action="store_true", help="Full HD 1920x1080 rendering")
    ap.add_argument("--res", default="", help="custom render size, e.g. 1600x900")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--touch", action="store_true", help="always show touch buttons")
    ap.add_argument("--mute", action="store_true")
    ap.add_argument("--fps", action="store_true")
    ap.add_argument("--maxfps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--demo", action="store_true", help="autopilot (attract mode)")
    ap.add_argument("--frames", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--shot", default="", help=argparse.SUPPRESS)
    ap.add_argument("--shot-at", dest="shot_at", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()
    game = Game(args)
    game.run()
    if args.frames:
        print("score", game.score, "wave", game.wave, "kills", game.kills, "hp", int(game.player.hp),
              "state", game.state, "enemies", len(game.enemies))
    pygame.quit()


if __name__ == "__main__":
    main()
