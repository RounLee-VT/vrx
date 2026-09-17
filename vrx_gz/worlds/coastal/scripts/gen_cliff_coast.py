#!/usr/bin/env python3
"""Sea-cliff coast with repeat-survey epochs (cf. proposal Fig. 1).

A ~30 m layered sandstone cliff with a wave-cut notch, sea caves, gullies and
sea stacks standing on a jointed wave-cut rock platform, a pocket beach in a
cove, submerged rock ridges and a sandy seabed. Unlike the heightfield worlds,
the cliff face and stacks are true 3D meshes, so overhangs (notch, caves) are
represented and visible to LiDAR / sonar.

Epochs (repeat surveys):
  0  baseline
  1  post-storm: block rockfall (face retreat + talus cone), notch deepening,
     top of the slender stack lost, pocket-beach sand loss
  2  later: cave enlargement, second rockfall, talus reworked,
     slender stack collapses to a stump, beach partly recovers

Outputs per epoch
  models/cliff_coast_terrain_epoch{e}/        ground, plateau, cliff+stacks
  worlds/cliff_coast_epoch{e}.sdf
  worlds/coastal/ground_truth/cliff_coast_epoch{e}.*   (json, ground/plateau
      DEMs, cliff-face offset map r(x, z), stack geometry, mesh points)
  worlds/coastal/ground_truth/cliff_change_epoch{a}_to_epoch{b}.*

Cliff-face parametrisation: a face point at along-shore x and height z is
    P = (x, y_c(x)) + r(x, z) * n(x),     n = seaward unit normal of y_c,
so r < 0 is landward (notch, cave, rockfall scar). Face retreat between
epochs is exactly dr = r_b - r_a.
"""

import argparse
import json
import math
import os

import numpy as np
from PIL import Image
from scipy import ndimage

import terrain_lib as tl
from gen_synthetic import GT, MODELS, WORLDS, fixed_noise, smoothstep

ANCHOR = (38.1700, -76.8700, 0.0)   # nominal, Potomac River bluffs area, VA

X0, X1, Y0, Y1, DX = -250.0, 250.0, -150.0, 250.0, 1.0
FACE_DS, FACE_DZ, Z_BOT = 0.5, 0.5, -3.0
TEX_LEN = 800.0          # metres of face texture (cliff 500 m + stacks)
TEX_ZMIN, TEX_ZMAX = -4.0, 38.0

CAVES = [  # x, half width, height, depth, floor z
    dict(x=-40.0, hw=8.0, h=7.0, depth=12.0, floor=0.3),
    dict(x=150.0, hw=5.0, h=5.0, depth=7.0, floor=0.4),
    dict(x=20.0, hw=4.0, h=4.0, depth=5.0, floor=0.5),
]
STACKS = [  # x, offshore distance from cliff line, base radius, height
    dict(x=60.0, off=30.0, r=7.0, h=22.0, seed=1),
    dict(x=95.0, off=48.0, r=4.5, h=16.0, seed=2),   # slender, collapses
    dict(x=-70.0, off=36.0, r=4.5, h=12.0, seed=3),
    dict(x=180.0, off=34.0, r=9.0, h=25.0, seed=4),
]
SLENDER = 1

EPOCHS = {
    0: dict(label='Survey 1 baseline', notch_extra=0.0,
            rockfalls=[], talus=[], cave0_extra=0.0, cave0_h_extra=0.0,
            stack_h={}, rubble={}, beach_dz=0.0),
    1: dict(label='Survey 2 post-storm: rockfall, notch deepening, stack top loss',
            notch_extra=0.8,
            rockfalls=[dict(x0=40.0, x1=75.0, z_low=6.0, retreat=4.5)],
            talus=[dict(x=57.0, off=7.0, h=3.0, rad=12.0)],
            cave0_extra=0.0, cave0_h_extra=0.0,
            stack_h={SLENDER: 9.0}, rubble={SLENDER: 1.2}, beach_dz=-0.6),
    2: dict(label='Survey 3: cave growth, 2nd rockfall, stack collapse, beach recovery',
            notch_extra=1.1,
            rockfalls=[dict(x0=40.0, x1=75.0, z_low=6.0, retreat=4.5),
                       dict(x0=-185.0, x1=-165.0, z_low=10.0, retreat=2.5)],
            talus=[dict(x=57.0, off=9.0, h=1.8, rad=16.0),
                   dict(x=-175.0, off=5.0, h=1.5, rad=7.0)],
            cave0_extra=3.0, cave0_h_extra=2.0,
            stack_h={SLENDER: 3.0}, rubble={SLENDER: 1.8}, beach_dz=-0.3),
}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def cliff_line(x):
    y = (8.0 * np.sin(2 * np.pi * x / 310.0 + 0.7) + 4.0 * np.sin(2 * np.pi * x / 97.0)
         - 38.0 * np.exp(-((x + 120.0) / 38.0) ** 2)      # cove
         + 18.0 * np.exp(-((x - 110.0) / 45.0) ** 2))     # headland
    dy = np.gradient(y, x) if np.ndim(x) and np.size(x) > 1 else 0.0 * y
    n = np.stack([-dy, np.ones_like(y)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return y, n


def plateau_z(x, y):
    big = 2.5 * np.sin(x / 55.0 + 0.3) * np.cos(y / 70.0) + 1.2 * np.sin(x / 23.0)
    return 30.0 + big + 0.015 * np.clip(-y, 0, None)


def cove_weight(x):
    return np.exp(-((x + 120.0) / 30.0) ** 2)


def face_offset(xs, zs, ztop, ep):
    """r(x, z) on the (nz, nx) face grid."""
    X, Z = np.meshgrid(xs, zs)
    rel = ztop[None, :] - Z
    r = 0.04 * rel                                              # slight batter
    layer = np.sin(2 * np.pi * Z / 3.1) + 0.5 * np.sin(2 * np.pi * Z / 1.3 + 0.8)
    r += 0.30 * layer                                           # hard/soft beds
    r += fixed_noise(X.shape, 31, (2.0, 6.0), 0.35)             # blocky relief
    r += fixed_noise(X.shape, 34, (40.0, 14.0), 1.8)            # buttresses
    r += 3.0 * np.sin(X / 23.0 + 1.1) * np.sin(X / 61.0)        # re-entrants
    rng = np.random.default_rng(32)
    for gx in rng.uniform(xs[0], xs[-1], 28):                   # gullies
        r -= rng.uniform(1.5, 4.0) * np.exp(-((X - gx) / rng.uniform(1.5, 4)) ** 2) \
            * smoothstep(-2, 6, Z)
    notch = (2.0 + ep['notch_extra']) * (1 - 0.7 * cove_weight(xs))[None, :]
    r -= notch * np.exp(-((Z - 1.3) / 1.4) ** 2)
    for i, c in enumerate(CAVES):
        depth = c['depth'] + (ep['cave0_extra'] if i == 0 else 0.0)
        h = c['h'] + (ep['cave0_h_extra'] if i == 0 else 0.0)
        a = ((X - c['x']) / c['hw']) ** 2 + (np.maximum(Z - c['floor'], 0) / h) ** 2
        w = (1 - smoothstep(0.7, 1.0, a)) * (Z > c['floor'] - 0.3)
        r -= depth * w
    for rf in ep['rockfalls']:
        wx = smoothstep(rf['x0'] - 4, rf['x0'], X) * (1 - smoothstep(rf['x1'], rf['x1'] + 4, X))
        scar = fixed_noise(X.shape, 33, (1.5, 3.0), 0.6)
        r -= (rf['retreat'] + scar) * wx * smoothstep(rf['z_low'] - 2, rf['z_low'] + 2, Z)
    return r


def build_face(ep):
    xs = np.arange(X0, X1 + 1e-6, FACE_DS)
    yc, n = cliff_line(xs)
    ztop = plateau_z(xs, yc) + 0.15
    nz = int(math.ceil((ztop.max() - Z_BOT) / FACE_DZ)) + 1
    t = np.linspace(0.0, 1.0, nz)
    Zg = Z_BOT + t[:, None] * (ztop - Z_BOT)[None, :]
    # r evaluated on the per-column z levels
    r = np.empty_like(Zg)
    zabs = np.arange(Z_BOT, ztop.max() + FACE_DZ, FACE_DZ / 2)
    rtab = face_offset(xs, zabs, ztop, ep)
    for j in range(len(xs)):
        r[:, j] = np.interp(Zg[:, j], zabs, rtab[:, j])
    P = np.stack([xs[None, :] + n[None, :, 0] * r,
                  yc[None, :] + n[None, :, 1] * r, Zg], -1)
    dPdz = np.gradient(P, axis=0)
    dPds = np.gradient(P, axis=1)
    N = np.cross(dPdz, dPds)
    UV = np.stack([np.broadcast_to((xs[None, :] - X0) / TEX_LEN, Zg.shape),
                   (Zg - TEX_ZMIN) / (TEX_ZMAX - TEX_ZMIN)], -1)
    face = dict(xs=xs, yc=yc, n=n, ztop=ztop, zabs=zabs, rtab=rtab, r=r, Z=Zg)
    return P, N, UV, face


def stack_geometry(k, st, ep, ground_fn):
    yc, n = cliff_line(np.array([st['x'] - 1, st['x'], st['x'] + 1]))
    cx = st['x'] + n[1, 0] * st['off']
    cy = yc[1] + n[1, 1] * st['off']
    zb = ground_fn(cx, cy) - 1.5
    h = ep['stack_h'].get(k, st['h'])
    ztop = zb + 1.5 + h
    nth = 72
    th = np.linspace(0, 2 * np.pi, nth + 1)
    nzs = max(4, int(math.ceil((ztop - zb) / 0.5)) + 1)
    zs = np.linspace(zb, ztop, nzs)
    TH, ZZ = np.meshgrid(th, zs)
    rng = np.random.default_rng(st['seed'])
    ph = rng.uniform(0, 2 * np.pi, 3)
    ang = (0.15 * np.sin(3 * TH + ph[0]) + 0.08 * np.sin(5 * TH + ph[1])
           + 0.05 * np.sin(9 * TH + ph[2]))
    nz = fixed_noise((nzs, nth), 40 + st['seed'], (2.0, 3.0), 0.10)
    nz = np.concatenate([nz, nz[:, :1]], axis=1)       # periodic seam
    taper = 1.0 - 0.18 * (ZZ - zb) / max(st['h'], 1.0)
    layer = 0.05 * np.sin(2 * np.pi * ZZ / 3.1)
    R = st['r'] * (1 + ang + nz + layer) * taper
    R -= 1.2 * np.exp(-((ZZ - 1.3) / 1.2) ** 2)          # notch
    R = np.maximum(R, 0.6)
    wob = 0.12 * st['r'] * np.array([np.sin((ZZ - zb) / 5.0 + ph[0]),
                                     np.cos((ZZ - zb) / 7.0 + ph[1])])
    P = np.stack([cx + wob[0] + R * np.cos(TH), cy + wob[1] + R * np.sin(TH), ZZ], -1)
    N = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))
    u0 = 510.0 + 70.0 * k
    arc = TH * st['r']
    UV = np.stack([(u0 + arc) / TEX_LEN, (ZZ - TEX_ZMIN) / (TEX_ZMAX - TEX_ZMIN)], -1)
    wall = tl.grid_part(P, N, UV)
    # rough top cap (fan)
    ring = P[-1, :-1]
    cx_top, cy_top = ring[:, 0].mean(), ring[:, 1].mean()
    top_noise = 0.6 * np.sin(4 * th[:-1] + ph[0]) if k in ep['stack_h'] else 0.0 * th[:-1]
    ring = ring.copy()
    ring[:, 2] += top_noise
    centre = np.array([[cx_top, cy_top, ztop + (0.2 if k not in ep['stack_h'] else -0.3)]])
    V = np.vstack([centre, ring])
    F = np.array([[0, 1 + i, 1 + (i + 1) % nth] for i in range(nth)])
    Nc = np.tile([0, 0, 1.0], (len(V), 1))
    UVc = np.column_stack([(u0 + 10) / TEX_LEN + 0 * V[:, 0],
                           np.full(len(V), (ztop - 1 - TEX_ZMIN) / (TEX_ZMAX - TEX_ZMIN))])
    cap = tl.mesh_part(V, Nc, UVc, F)
    # collapse-rubble is added to the ground; mean radius profile for GT
    info = dict(index=k, centre_xy=[float(cx), float(cy)], base_z=float(zb),
                top_z=float(ztop), height_above_bed=float(h),
                mean_radius_profile=dict(z=zs.tolist(), r=R.mean(axis=1).tolist()),
                volume_m3=float(np.trapezoid(np.pi * (R[:, :-1] ** 2).mean(axis=1), zs)))
    return [wall, cap], info


def build_ground(ep, face):
    xs = np.arange(X0, X1 + 1e-6, DX)
    ys = np.arange(Y1, Y0 - 1e-6, -DX)
    X, Y = np.meshgrid(xs, ys)
    yc, _ = cliff_line(xs)
    rmin = np.array([face['r'][:, j].min() for j in range(0, len(face['xs']), 2)])
    y_back = (yc + np.minimum(rmin, 0.0) - 3.0)[None, :]
    D = Y - yc[None, :]

    # jointed wave-cut platform -> sandy seabed
    z = 0.2 - 0.04 * np.clip(D, 0, None)
    z = np.where(D < 70, z, -2.6 - 11.4 * smoothstep(70, 280, D))
    joints = np.zeros_like(X)
    for ang, sp in ((np.radians(25), 11.0), (np.radians(-60), 17.0), (np.radians(80), 23.0)):
        u = (X * np.cos(ang) + Y * np.sin(ang)) / sp
        joints = np.maximum(joints, np.exp(-((u - np.round(u)) * sp / 0.7) ** 2))
    rocky = 1 - smoothstep(60, 90, D)
    z += rocky * (-0.45 * joints + fixed_noise(X.shape, 51, 3, 0.25)
                  + fixed_noise(X.shape, 52, 1, 0.08))
    for dc, hr in ((125.0, 1.3), (165.0, 0.9)):               # submerged ridges
        ridge = np.exp(-((D - dc - 7 * np.sin(X / 40.0)) / 4.0) ** 2)
        z += hr * ridge * (1 + fixed_noise(X.shape, 53, 4, 0.3))
    z += (1 - rocky) * fixed_noise(X.shape, 54, 2, 0.05)       # sand ripples
    z = np.where(D < 0, 0.3 + fixed_noise(X.shape, 55, 2, 0.15), z)

    # pocket beach in the cove
    wc = cove_weight(X)
    beach = 2.2 + ep['beach_dz'] * smoothstep(-5, 5, D) * (1 - smoothstep(30, 45, D)) \
        - 0.07 * np.clip(D, -10, None)
    beach_on = (wc > 0.05) & (D < 60)
    z = np.where(beach_on, np.maximum(z, z * (1 - wc) + beach * wc), z)

    # talus cones and stack rubble
    talus = np.zeros_like(X, bool)
    for tc in ep['talus']:
        ycx, nn = cliff_line(np.array([tc['x'] - 1, tc['x'], tc['x'] + 1]))
        cx, cy = tc['x'] + nn[1, 0] * tc['off'], ycx[1] + nn[1, 1] * tc['off']
        rr = np.hypot(X - cx, Y - cy)
        bump = tc['h'] * np.exp(-(rr / tc['rad']) ** 2) * (1 + fixed_noise(X.shape, 56, 0.8, 0.25))
        z += np.clip(bump, 0, None)
        talus |= bump > 0.15
    stacks_xy = []
    for k, st in enumerate(STACKS):
        ycx, nn = cliff_line(np.array([st['x'] - 1, st['x'], st['x'] + 1]))
        cx, cy = st['x'] + nn[1, 0] * st['off'], ycx[1] + nn[1, 1] * st['off']
        stacks_xy.append((cx, cy))
        if k in ep['rubble']:
            rr = np.hypot(X - cx, Y - cy)
            bump = ep['rubble'][k] * np.exp(-((rr - st['r']) / 4.0) ** 2) \
                * (1 + fixed_noise(X.shape, 57, 0.8, 0.35)) * (rr > st['r'] * 0.6)
            z += np.clip(bump, 0, None)
            talus |= bump > 0.15

    masks = {
        'rock': (rocky > 0.5) & ~beach_on & ~talus & (z > -6),
        'algae': (rocky > 0.5) & ~beach_on & ~talus & (z > -0.3) & (z < 0.5)
                 & (fixed_noise(X.shape, 58, 2, 1.0) > 0.4),
        'sand': beach_on & (wc > 0.3),
        'talus': talus,
    }
    return X, Y, z.astype(np.float32), y_back, masks, stacks_xy


def build_plateau(face):
    xs = np.arange(X0, X1 + 1e-6, DX)
    ys = np.arange(Y1, Y0 - 1e-6, -DX)
    X, Y = np.meshgrid(xs, ys)
    j = np.searchsorted(face['xs'], xs)
    # top edge = face top row position
    top = face['Z'].shape[0] - 1
    ex = face['xs'] + face['n'][:, 0] * face['r'][top]
    ey = face['yc'] + face['n'][:, 1] * face['r'][top]
    y_edge = np.interp(xs, ex, ey)[None, :]
    keep_v = Y <= y_edge
    Ys = np.minimum(Y, y_edge)
    Z = plateau_z(X, Ys)
    path = np.exp(-((Ys - (y_edge - 14 - 5 * np.sin(X / 30.0))) / 1.2) ** 2)
    Z = Z - 0.08 * path
    cell_keep = keep_v[:-1, :-1] | keep_v[1:, :-1] | keep_v[:-1, 1:] | keep_v[1:, 1:]
    masks = {'grass': np.ones_like(Z, bool),
             'bare': (path > 0.4) | ((y_edge - Ys < 2.5) & (fixed_noise(Z.shape, 61, 2, 1) > 0.3))}
    masks['grass'] &= ~masks['bare']
    return X, Ys, Z.astype(np.float32), cell_keep, keep_v, masks


# ---------------------------------------------------------------------------
# Textures
# ---------------------------------------------------------------------------

def cliff_texture(face, ep, path, W=8192, H=672):
    u = (np.arange(W) + 0.5) / W * TEX_LEN               # metres along texture
    z = TEX_ZMAX - (np.arange(H) + 0.5) / H * (TEX_ZMAX - TEX_ZMIN)
    U, Zt = np.meshgrid(u, z)
    rng = np.random.default_rng(71)

    # bedding: variable-thickness layers, gently warped, soft boundaries
    cols = np.array([(0.90, 0.83, 0.67), (0.84, 0.71, 0.53), (0.88, 0.74, 0.52),
                     (0.80, 0.74, 0.62), (0.83, 0.64, 0.46), (0.88, 0.85, 0.77),
                     (0.77, 0.64, 0.49)])
    thick = rng.uniform(0.35, 2.6, 80)
    bounds = np.concatenate([[TEX_ZMIN - 6], TEX_ZMIN - 6 + np.cumsum(thick)])
    layer_col = cols[rng.integers(0, len(cols), 80)]
    layer_col *= rng.uniform(0.92, 1.06, (80, 1))
    zz = Zt + 0.6 * np.sin(U / 41.0) + 0.3 * np.sin(U / 13.0 + 1.0) \
        + fixed_noise((H, W), 75, (2, 40), 0.25)
    idx = np.clip(np.searchsorted(bounds, zz) - 1, 0, 79)
    frac = (zz - bounds[idx]) / thick[idx]
    nxt = np.clip(idx + 1, 0, 79)
    edge = smoothstep(0.85, 1.0, frac)[..., None]
    rgb = layer_col[idx] * (1 - edge) + layer_col[nxt] * edge
    rgb *= (1 + 0.05 * fixed_noise((H, W), 72, (0.6, 25), 1.0))[..., None]   # lamination
    rgb *= (1 + 0.06 * fixed_noise((H, W), 74, 0.8, 1.0))[..., None]         # grain
    streak = fixed_noise((H, W), 73, (80, 2.5), 1.0)
    rgb *= (1 - 0.08 * np.clip(streak, 0, None))[..., None]                  # runoff stains

    # cliff part: bake occlusion from the face geometry, fresh scars, soil lip
    x_face = u + X0
    in_face = x_face <= X1
    jx = np.interp(x_face, face['xs'], np.arange(len(face['xs'])))
    iz = np.interp(z, face['zabs'], np.arange(len(face['zabs'])))
    JX, IZ = np.meshgrid(jx, iz)
    r = ndimage.map_coordinates(face['rtab'], [IZ, JX], order=1, mode='nearest')
    r_ref = ndimage.map_coordinates(
        ndimage.maximum_filter(ndimage.gaussian_filter(face['rtab'], (6, 6)), (60, 40)),
        [IZ, JX], order=1, mode='nearest')
    occ = np.clip((r_ref - r - 2.0) / 6.0, 0, 1) * in_face[None, :]
    rgb *= (1 - 0.75 * occ ** 1.2)[..., None]
    for rf in ep['rockfalls']:
        wx = smoothstep(rf['x0'] - 2, rf['x0'] + 2, U + X0) * \
            (1 - smoothstep(rf['x1'] - 2, rf['x1'] + 2, U + X0))
        fresh = wx * smoothstep(rf['z_low'] - 1, rf['z_low'] + 2, Zt) * in_face[None, :]
        rgb = rgb * (1 - 0.5 * fresh[..., None]) + \
            np.array([0.93, 0.85, 0.68]) * (0.5 * fresh[..., None])

    wet = 1 - smoothstep(0.8, 2.6, Zt + 0.3 * fixed_noise((H, W), 76, (2, 8), 1.0))
    rgb *= (1 - 0.4 * wet)[..., None]
    algae = 1 - smoothstep(-0.2, 0.9, Zt)
    rgb = rgb * (1 - 0.8 * algae[..., None]) + np.array([0.23, 0.25, 0.17]) * 0.8 * algae[..., None]
    ztop_u = np.interp(u, face['xs'] - X0, face['ztop'], left=1e9, right=1e9)
    dtop = Zt - ztop_u[None, :]
    soil = smoothstep(-1.4, -0.4, dtop)
    rgb = rgb * (1 - soil[..., None]) + np.array([0.36, 0.29, 0.20]) * soil[..., None]
    grass = smoothstep(-0.4, -0.1, dtop)
    rgb = rgb * (1 - grass[..., None]) + np.array([0.38, 0.45, 0.27]) * grass[..., None]
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)


def plateau_texture(Z, masks, path):
    sh = Z.shape
    g = fixed_noise(sh, 91, 6, 1.0)
    rgb = tl._lerp((0.42, 0.48, 0.28), (0.30, 0.40, 0.22), smoothstep(-1, 1, g))
    dry = smoothstep(0.3, 1.2, fixed_noise(sh, 92, 12, 1.0))
    rgb = rgb * (1 - 0.6 * dry[..., None]) + np.array([0.60, 0.58, 0.40]) * 0.6 * dry[..., None]
    shrubs = fixed_noise(sh, 93, 1.2, 1.0) > 1.3
    rgb[shrubs] = rgb[shrubs] * 0.55
    bare = masks['bare']
    rgb[bare] = np.array([0.66, 0.59, 0.45]) * (1 + 0.08 * fixed_noise(sh, 94, 1, 1.0))[bare][:, None]
    rgb *= (1 + 0.07 * fixed_noise(sh, 95, 0.7, 1.0))[..., None]
    rgb *= (0.7 + 0.3 * tl.hillshade(Z, DX))[..., None]
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    img.resize((sh[1] * 2, sh[0] * 2), Image.BICUBIC).save(path)


# ---------------------------------------------------------------------------
# Model / world / ground truth
# ---------------------------------------------------------------------------

MODEL_SDF = """<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <link name="link">
{visuals}
{collisions}
    </link>
  </model>
</sdf>
"""


def visual(name, uri):
    return f"""      <visual name="{name}">
        <cast_shadows>{'true' if name == 'cliff' else 'false'}</cast_shadows>
        <geometry><mesh><uri>{uri}</uri></mesh></geometry>
      </visual>"""


def collision(name, uri):
    return f"""      <collision name="{name}">
        <geometry><mesh><uri>{uri}</uri></mesh></geometry>
        <surface><friction><ode><mu>0.8</mu><mu2>0.8</mu2></ode></friction></surface>
      </collision>"""


def generate_epoch(e, ep, sea_state):
    world = f'cliff_coast_epoch{e}'
    name = f'cliff_coast_terrain_epoch{e}'
    root = os.path.join(MODELS, name)
    mdir = os.path.join(root, 'meshes')
    os.makedirs(mdir, exist_ok=True)
    print(f'[{world}] {ep["label"]}')

    # --- cliff face
    Pf, Nf, UVf, face = build_face(ep)
    # --- ground (platform/seabed/beach), snapped landward edge
    X, Y, Zg, y_back, gmasks, _ = build_ground(ep, face)
    gx = X[0]
    ground_fn = lambda x, y: float(ndimage.map_coordinates(  # noqa: E731
        Zg, [[(Y1 - y) / DX], [(x - X0) / DX]], order=1)[0])
    Ys = np.maximum(Y, y_back)
    keep_g = Y >= y_back
    cell_g = keep_g[:-1, :-1] | keep_g[1:, :-1] | keep_g[:-1, 1:] | keep_g[1:, 1:]
    Zs = np.array([np.interp(Ys[:, j], Y[::-1, j], Zg[::-1, j]) for j in range(Y.shape[1])]).T
    Pg = np.stack([X, Ys, Zs], -1)
    Ng = tl.heightfield_normals(Zg, DX)
    UVg = np.stack([(X - X0) / (X1 - X0), (Y - Y0) / (Y1 - Y0)], -1)
    rgb = tl.colorize(Zg, DX, gmasks, seed=81)
    Image.fromarray(rgb).resize((rgb.shape[1] * 2, rgb.shape[0] * 2), Image.BICUBIC) \
        .save(os.path.join(mdir, 'ground.png'))
    nv, nt = tl.write_parts_obj(os.path.join(mdir, 'ground.obj'),
                                tl.split_grid(Pg, Ng, UVg, cell_g), 'ground.png')
    s = 3
    cnv, cnt = tl.write_parts_obj(
        os.path.join(mdir, 'ground_collision.obj'),
        tl.split_grid(Pg[::s, ::s], Ng[::s, ::s], None,
                      (keep_g[::s, ::s][:-1, :-1] | keep_g[::s, ::s][1:, 1:])
                      & (Zg[::s, ::s][:-1, :-1] > -4)))
    print(f'  ground  {nv} v / {nt} t   (collision {cnt} t)')

    # --- plateau
    Xp, Yp, Zp, cell_p, keep_p, pmasks = build_plateau(face)
    Pp = np.stack([Xp, Yp, Zp], -1)
    Np = tl.heightfield_normals(Zp, DX)
    UVp = np.stack([(Xp - X0) / (X1 - X0), (Y - Y0) / (Y1 - Y0)], -1)
    plateau_texture(Zp, pmasks, os.path.join(mdir, 'plateau.png'))
    nv, nt = tl.write_parts_obj(os.path.join(mdir, 'plateau.obj'),
                                tl.split_grid(Pp, Np, UVp, cell_p), 'plateau.png')
    print(f'  plateau {nv} v / {nt} t')

    # --- cliff face + stacks (shared strata texture)
    cliff_texture(face, ep, os.path.join(mdir, 'cliff.png'))
    parts = tl.split_grid(Pf, Nf, UVf, None, block=256)
    stack_info, stack_parts = [], []
    for k, st in enumerate(STACKS):
        p, info = stack_geometry(k, st, ep, ground_fn)
        stack_parts += p
        stack_info.append(info)
    nv, nt = tl.write_parts_obj(os.path.join(mdir, 'cliff.obj'), parts + stack_parts,
                                'cliff.png')
    cparts = tl.split_grid(Pf[::2, ::2], Nf[::2, ::2], None, None, block=256)
    for k, st in enumerate(STACKS):
        p, _ = stack_geometry(k, st, ep, ground_fn)
        cparts += p
    ccnv, ccnt = tl.write_parts_obj(os.path.join(mdir, 'cliff_collision.obj'), cparts)
    print(f'  cliff   {nv} v / {nt} t   (collision {ccnt} t)')

    with open(os.path.join(root, 'model.config'), 'w') as f:
        f.write(f"""<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author><name>vrx_gz/worlds/coastal/scripts</name></author>
  <description>Sea-cliff coast terrain, {ep['label']}</description>
</model>
""")
    with open(os.path.join(root, 'model.sdf'), 'w') as f:
        f.write(MODEL_SDF.format(
            name=name,
            visuals='\n'.join([visual('ground', 'meshes/ground.obj'),
                               visual('plateau', 'meshes/plateau.obj'),
                               visual('cliff', 'meshes/cliff.obj')]),
            collisions='\n'.join([collision('ground', 'meshes/ground_collision.obj'),
                                  collision('cliff', 'meshes/cliff_collision.obj')])))

    lat, lon, elev = ANCHOR
    waves = tl.wave_params(sea_state, -math.pi / 2)
    tl.write_world(os.path.join(WORLDS, f'{world}.sdf'), world, lat, lon, elev,
                   [tl.include(name, 'terrain')], waves,
                   camera_pose='40 230 55 0 0.32 -1.75',
                   comment=f'{ep["label"]}. Synthetic ~30 m sea cliff with notch, caves, '
                           'stacks, wave-cut platform (cf. proposal Fig. 1).')

    # --- ground truth
    info = {
        'world': world, 'epoch': e, 'epoch_params': ep,
        'frame': 'Gazebo world ENU (x east, y north, z up), metres; z=0 still water',
        'origin_lat_deg': lat, 'origin_lon_deg': lon, 'sea_state': sea_state,
        'caves': CAVES, 'stacks': stack_info,
        'face_param': 'P = (x, y_c(x)) + r(x,z) * n(x); arrays in *_face.npz',
        'ground_grid': {'dx_m': DX, 'x_west_m': X0, 'y_north_m': Y1,
                        'note': 'z[row,col] at x=x_west+col*dx, y=y_north-row*dx; '
                                'valid where mask_valid (seaward of the cliff base)'},
        'plateau_grid': 'same grid as ground; valid where mask_valid (landward of top edge)',
        'meshes': f'vrx_gz/models/{name}/meshes/{{ground,plateau,cliff}}.obj '
                  '(exact simulated geometry)',
    }
    base = os.path.join(GT, world)
    with open(base + '.json', 'w') as f:
        json.dump(info, f, indent=2)
    np.savez_compressed(base + '_ground.npz', z=Zg, mask_valid=keep_g,
                        **{f'mask_{k}': v for k, v in gmasks.items()})
    np.savez_compressed(base + '_plateau.npz', z=np.where(keep_p, Zp, np.nan).astype(np.float32),
                        mask_valid=keep_p)
    np.savez_compressed(base + '_face.npz', x=face['xs'], z=face['zabs'],
                        r=face['rtab'].astype(np.float32), z_top=face['ztop'],
                        y_c=face['yc'], normal=face['n'])
    pts = np.vstack([Pg[keep_g].reshape(-1, 3), Pp[keep_p].reshape(-1, 3),
                     Pf.reshape(-1, 3)])
    np.savez_compressed(base + '_mesh_points.npz', xyz=pts.astype(np.float32))
    save_overview(base + '.png', world, ep, Zg, keep_g, Zp, keep_p, face, stack_info)
    return dict(Zg=Zg, keep_g=keep_g, face=face, stacks=stack_info)


def save_overview(path, world, ep, Zg, keep_g, Zp, keep_p, face, stacks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    top = np.where(keep_p, Zp, np.where(keep_g, Zg, np.nan))
    fig, axs = plt.subplots(1, 2, figsize=(18, 7))
    im = axs[0].imshow(top, cmap='terrain', vmin=-14, vmax=34,
                       extent=[X0, X1, Y0, Y1])
    for st in stacks:
        axs[0].add_patch(plt.Circle(st['centre_xy'], np.mean(st['mean_radius_profile']['r']),
                                    color='saddlebrown'))
    axs[0].set_title(f'{world}: top surface (plateau / ground / stacks)')
    plt.colorbar(im, ax=axs[0], shrink=0.8, label='z [m]')
    im2 = axs[1].imshow(face['rtab'], origin='lower', aspect='auto', cmap='RdBu',
                        vmin=-12, vmax=12,
                        extent=[face['xs'][0], face['xs'][-1], face['zabs'][0], face['zabs'][-1]])
    axs[1].plot(face['xs'], face['ztop'], 'k', lw=0.8)
    axs[1].set_title('cliff face offset r(x, z) [m] (red = landward: notch, caves, scars)')
    axs[1].set_xlabel('x [m]')
    axs[1].set_ylabel('z [m]')
    plt.colorbar(im2, ax=axs[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def change_products(res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = {}
    for a, b in [(0, 1), (1, 2), (0, 2)]:
        A, B = res[a], res[b]
        key = f'cliff_change_epoch{a}_to_epoch{b}'
        valid = A['face']['zabs'][:, None] <= np.minimum(A['face']['ztop'], B['face']['ztop'])[None, :]
        dr = np.where(valid, B['face']['rtab'] - A['face']['rtab'], np.nan)
        both = A['keep_g'] & B['keep_g']
        dz = np.where(both, B['Zg'] - A['Zg'], np.nan)
        np.savez_compressed(os.path.join(GT, key + '.npz'), face_dr=dr.astype(np.float32),
                            face_x=A['face']['xs'], face_z=A['face']['zabs'],
                            ground_dz=dz.astype(np.float32))
        cell = FACE_DS * (FACE_DZ / 2)
        stacks = [{'index': i, 'height_change_m': sb['height_above_bed'] - sa['height_above_bed'],
                   'volume_change_m3': sb['volume_m3'] - sa['volume_m3']}
                  for i, (sa, sb) in enumerate(zip(A['stacks'], B['stacks']))]
        summary[key] = {
            'face_retreat_volume_m3': float(-np.nansum(np.where(dr < -0.02, dr, 0)) * cell),
            'face_max_retreat_m': float(-np.nanmin(dr)),
            'ground_deposition_m3': float(np.nansum(np.where(dz > 0.02, dz, 0)) * DX * DX),
            'ground_erosion_m3': float(-np.nansum(np.where(dz < -0.02, dz, 0)) * DX * DX),
            'stacks': [s for s in stacks if abs(s['height_change_m']) > 1e-6],
        }
        fig, axs = plt.subplots(1, 2, figsize=(18, 6.5))
        im = axs[0].imshow(dr, origin='lower', aspect='auto', cmap='RdBu', vmin=-6, vmax=6,
                           extent=[A['face']['xs'][0], A['face']['xs'][-1],
                                   A['face']['zabs'][0], A['face']['zabs'][-1]])
        axs[0].set_title(f'{key}: cliff face dr [m] (red = retreat)')
        axs[0].set_xlabel('x [m]')
        axs[0].set_ylabel('z [m]')
        plt.colorbar(im, ax=axs[0], shrink=0.8)
        im = axs[1].imshow(dz, cmap='RdBu', vmin=-2, vmax=2, extent=[X0, X1, Y0, Y1])
        axs[1].set_title('ground dz [m] (blue = deposition: talus, rubble)')
        plt.colorbar(im, ax=axs[1], shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(GT, key + '.png'), dpi=80)
        plt.close(fig)
    with open(os.path.join(GT, 'cliff_change_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sea-state', default='moderate', choices=tl.SEA_STATES)
    p.add_argument('--epochs', type=int, nargs='*', default=sorted(EPOCHS))
    args = p.parse_args()
    os.makedirs(GT, exist_ok=True)
    res = {e: generate_epoch(e, EPOCHS[e], args.sea_state) for e in args.epochs}
    if all(e in res for e in (0, 1, 2)):
        change_products(res)


if __name__ == '__main__':
    main()
