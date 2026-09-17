#!/usr/bin/env python3
"""Synthetic worlds with exact ground truth.

1. claytor_lake_calm
   Still-water reservoir cove (stand-in for the Claytor Lake test in
   Milestone 1). Contains geometric calibration targets (flat seabed patch,
   submerged boxes/sphere, pilings, shore landmarks) whose exact poses are
   exported for mapping-accuracy evaluation.

2. nbs_surf_epoch{0,1,2}
   Hampton-Roads-like sandy bay shoreline with three alongshore sections:
     A (west)  living shoreline: marsh platform + tidal creek + oyster-reef sills
     B (mid)   unprotected control beach with dune and rhythmic sandbar
     C (east)  grey infrastructure: detached rock breakwaters with salients
   Epochs model repeat surveys (Oct 2026 / Jan 2027 / Mar 2027): storm
   erosion, bar migration, lee deposition, sill-gap scour, sill damage and
   partial recovery. Every epoch shares the same fine-scale texture/noise, so
   the exported difference maps contain only the prescribed morphology change.

Usage:  python3 gen_synthetic.py [--sea-state moderate] [--only lake|nbs]
"""

import argparse
import json
import math
import os

import numpy as np
from scipy import ndimage

import terrain_lib as tl

# scripts live in vrx_gz/worlds/coastal/scripts
VRX_GZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODELS = os.path.join(VRX_GZ, 'models')
WORLDS = os.path.join(VRX_GZ, 'worlds')
CONFIG = os.path.join(VRX_GZ, 'config')
GT = os.path.join(WORLDS, 'coastal', 'ground_truth')

# Nominal geodetic anchors (used for NavSat / GeoTIFF only).
CLAYTOR_LAKE = (37.0580, -80.6260, 257.9)   # approx. pool elev. 846 ft NGVD
HAMPTON_ROADS = (36.9320, -76.1900, 0.0)    # Norfolk bay shoreline area


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def fixed_noise(shape, seed, sigma_cells, amp):
    rng = np.random.default_rng(seed)
    n = ndimage.gaussian_filter(rng.standard_normal(shape), sigma_cells)
    return amp * n / (n.std() + 1e-12)


def bilinear(Z, dx, x_west, y_north, x, y):
    c = (x - x_west) / dx
    r = (y_north - y) / dx
    return float(ndimage.map_coordinates(Z, [[r], [c]], order=1)[0])


# ===========================================================================
# 1. Claytor Lake (still water)
# ===========================================================================

def nearest_where(cond, X, Y, seed):
    """World (x, y) of the True cell in `cond` closest to `seed`."""
    d2 = np.where(cond, (X - seed[0]) ** 2 + (Y - seed[1]) ** 2, np.inf)
    i = np.unravel_index(np.argmin(d2), d2.shape)
    return float(X[i]), float(Y[i]), i


def build_lake():
    dx = 1.0
    x_west, x_east, y_south, y_north = -300.0, 300.0, -240.0, 240.0
    nx = int(round((x_east - x_west) / dx)) + 1
    ny = int(round((y_north - y_south) / dx)) + 1
    X, Y = np.meshgrid(x_west + dx * np.arange(nx), y_north - dx * np.arange(ny))

    # Irregular cove outline (open to the west like a reservoir arm).
    th = np.arctan2(Y / 170.0, X / 230.0)
    rho = np.hypot(X / 230.0, Y / 170.0)
    r_edge = (1.0 + 0.10 * np.sin(3 * th + 0.4) + 0.06 * np.sin(5 * th + 1.3)
              + 0.04 * np.sin(9 * th))
    water = (rho < r_edge) | ((X < -150) & (np.abs(Y + 10 * np.sin(X / 40)) < 70))
    water = ndimage.binary_opening(water, iterations=3)

    d_in = ndimage.distance_transform_edt(water) * dx
    d_out = ndimage.distance_transform_edt(~water) * dx

    # Rock bluff along the north-west shore (steep bank above and below water).
    wb = smoothstep(-240, -210, X) * (1 - smoothstep(-70, -40, X)) * smoothstep(20, 60, Y)

    # Reservoir bank ~1:8 flattening to a ~11 m floor (steeper under the
    # bluff), plus a drowned river channel (thalweg) along the cove axis.
    z_water = -11.0 * (1.0 - np.exp(-d_in / (80.0 * (1 - 0.7 * wb))))
    chan = np.abs(Y - (-20 + 25 * np.sin((X + 300) / 110.0)))
    z_water -= 4.0 * np.exp(-(chan / 20.0) ** 2) * smoothstep(20, 60, d_in)

    # Valley-side hills rising ~35-45 m above pool (Appalachian relief).
    hills = (fixed_noise((ny, nx), 11, 30, 7.0) + fixed_noise((ny, nx), 12, 8, 1.2)
             + fixed_noise((ny, nx), 13, 3, 0.3))
    ridge = 36.0 * smoothstep(5, 170, d_out)
    z_land = np.minimum(0.35 * d_out, 3.0) + ridge + hills * smoothstep(5, 50, d_out)
    bluff = np.minimum(2.6 * d_out, 24.0 + 0.2 * d_out) + hills * smoothstep(15, 60, d_out)
    z_land = z_land * (1 - wb) + np.maximum(z_land, bluff) * wb
    Z = np.where(water, z_water, z_land)
    Z = ndimage.gaussian_filter(Z, 1.0)
    extra = {}

    # Flat seabed calibration patch: exactly -6.000 m, 40 m x 40 m, placed
    # where the natural bed is ~6 m deep so the blend ramps stay gentle.
    fx, fy, _ = nearest_where(np.abs(ndimage.uniform_filter(Z, 41) + 6.0) < 0.2,
                              X, Y, (40.0, 60.0))
    ddx = np.abs(X - fx) - 20.0
    ddy = np.abs(Y - fy) - 20.0
    w = 1.0 - smoothstep(0.0, 15.0, np.maximum(ddx, ddy))
    Z = Z * (1 - w) - 6.0 * w
    Z[(ddx <= 0) & (ddy <= 0)] = -6.0
    extra.update(flat_patch_center=[fx, fy], flat_patch_size=40.0,
                 flat_patch_z=-6.0)

    # Concrete boat ramp (1:8) on the north-east shore, running downslope.
    gy, gx = np.gradient(ndimage.gaussian_filter(Z, 4), dx)
    sx, sy, si = nearest_where(np.abs(Z) < 0.3, X, Y, (170.0, 130.0))
    down = np.array([-gx[si], gy[si]])          # toward the water (ENU)
    down /= np.linalg.norm(down)
    s = (X - sx) * down[0] + (Y - sy) * down[1]
    t = (X - sx) * -down[1] + (Y - sy) * down[0]
    on_ramp = (np.abs(t) < 3.0) & (s > -14) & (s < 24)
    Z[on_ramp] = np.clip(-s[on_ramp] / 8.0, -3.0, 1.75)
    extra.update(ramp_top_xy=[sx - 14 * down[0], sy - 14 * down[1]],
                 ramp_dir_xy=down.tolist(), ramp_slope='1:8')

    gy_, gx_ = np.gradient(Z, dx)
    slope = np.hypot(gx_, gy_)
    rock = (slope > 0.9) & (Z > -1.0) & ~on_ramp
    masks = {
        'grass': (Z > 1.0) & (Z <= 8.0) & ~on_ramp & ~rock,
        'forest': (Z > 8.0) & ~rock,
        'rock': rock,
        'concrete': on_ramp,
    }
    grid = dict(dx=dx, x_west=x_west, y_north=y_north, X=X, Y=Y,
                downslope=(gx, gy))
    return Z, masks, grid, extra


def lake_surround(Z, g, dx=10.0, half_x=1500.0, half_y=1200.0):
    """Coarse (10 m) valley terrain around the detailed lake area, continuing
    the reservoir arm westward. Overlaps the inner area by one cell, 0.3 m
    below it, so there are no visible seams."""
    xs = np.arange(-half_x, half_x + 1e-6, dx)
    ys = np.arange(half_y, -half_y - 1e-6, -dx)
    X, Y = np.meshgrid(xs, ys)
    ix0, iy1 = g['x_west'], g['y_north']
    ny, nx = Z.shape
    ix1, iy0 = ix0 + (nx - 1) * g['dx'], iy1 - (ny - 1) * g['dx']
    cx, cy = np.clip(X, ix0, ix1), np.clip(Y, iy0, iy1)
    z_edge = ndimage.map_coordinates(Z, [(iy1 - cy) / g['dx'], (cx - ix0) / g['dx']],
                                     order=1)
    dist = np.hypot(X - cx, Y - cy)
    hills = 28.0 + fixed_noise(X.shape, 14, 6, 14.0) + fixed_noise(X.shape, 15, 1.5, 2.0)
    w = smoothstep(0, 250, dist)
    zs = z_edge * (1 - w) + hills * w
    arm_c = -10 * np.sin(X / 40.0) + 60 * np.sin((X + 300) / 400.0)
    arm = (1 - smoothstep(50, 80, np.abs(Y - arm_c))) * (X < ix0 + 5)
    zs = zs * (1 - arm) + np.minimum(zs, -9.0) * arm
    inside = (X >= ix0) & (X <= ix1) & (Y >= iy0) & (Y <= iy1)
    zs = np.where(inside, z_edge - 0.3, zs)
    shrink = (X > ix0 + dx) & (X < ix1 - dx) & (Y > iy0 + dx) & (Y < iy1 - dx)
    cell_mask = ~(shrink[:-1, :-1] & shrink[1:, :-1] & shrink[:-1, 1:] & shrink[1:, 1:])
    masks = {'grass': (zs > 1.0) & (zs <= 8.0), 'forest': zs > 8.0}
    return dict(Z=zs.astype(np.float32), dx=dx, x_west=-half_x, y_north=half_y,
                cell_mask=cell_mask, masks=masks)


def box_model(name, pose, size, rgba, collide=True):
    sx, sy, sz = size
    col = f"""
        <collision name="collision"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>""" if collide else ''
    return f"""    <model name="{name}">
      <static>true</static>
      <pose>{pose}</pose>
      <link name="link">
        <visual name="visual">
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
          <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>
        </visual>{col}
      </link>
    </model>"""


def cyl_model(name, pose, radius, length, rgba):
    return f"""    <model name="{name}">
      <static>true</static>
      <pose>{pose}</pose>
      <link name="link">
        <visual name="visual">
          <geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>
          <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>
        </visual>
        <collision name="collision"><geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry></collision>
      </link>
    </model>"""


def sphere_model(name, pose, radius, rgba):
    return f"""    <model name="{name}">
      <static>true</static>
      <pose>{pose}</pose>
      <link name="link">
        <visual name="visual">
          <geometry><sphere><radius>{radius}</radius></sphere></geometry>
          <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>
        </visual>
        <collision name="collision"><geometry><sphere><radius>{radius}</radius></sphere></geometry></collision>
      </link>
    </model>"""


def generate_lake(args):
    print('[claytor_lake_calm]')
    Z, masks, g, extra = build_lake()
    name = 'claytor_lake_terrain'
    tl.write_terrain_model(MODELS, name, Z, g['dx'], g['x_west'], g['y_north'],
                           masks, description='Synthetic still-water reservoir '
                           'cove with calibration targets', seed=1,
                           surround=lake_surround(Z, g))

    bed = lambda x, y: bilinear(Z, g['dx'], g['x_west'], g['y_north'], x, y)
    targets, snippets = [], []

    def add(kind, name, x, y, z, yaw=0.0, **geom):
        pose = f'{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.4f}'
        targets.append({'name': name, 'type': kind,
                        'pose_xyz_yaw': [x, y, z, yaw], **geom})
        return pose

    X, Y = g['X'], g['Y']
    gx, gy = g['downslope']

    # Submerged boxes (2 x 2 x 1 m) resting on the bed at 3 / 6 / 10 m depth.
    flat = np.hypot(gx, gy) < 0.2
    fx, fy = extra['flat_patch_center']
    off_patch = np.hypot(X - fx, Y - fy) > 45
    for i, (depth, seed) in enumerate([(3.0, (-60.0, 110.0)),
                                       (6.0, (-20.0, -120.0)),
                                       (10.0, (-120.0, -40.0))]):
        x, y, _ = nearest_where((np.abs(Z + depth) < 0.1) & flat & off_patch,
                                X, Y, seed)
        zb = bed(x, y)
        n = f'gt_box_{i}'
        snippets.append(box_model(n, add('box', n, x, y, zb + 0.5, 0.5 * i,
                                         size=[2, 2, 1], bed_z=zb),
                                  (2, 2, 1), '0.9 0.5 0.1 1'))
    # Sphere and box on the flat patch.
    snippets.append(sphere_model('gt_sphere', add('sphere', 'gt_sphere', fx - 10,
                                                  fy, -6.0 + 1.0, radius=1.0),
                                 1.0, '0.8 0.8 0.1 1'))
    snippets.append(box_model('gt_box_flat', add('box', 'gt_box_flat', fx + 10, fy,
                                                 -6.0 + 0.75, 0.3,
                                                 size=[3, 1.5, 1.5]),
                              (3, 1.5, 1.5), '0.1 0.6 0.9 1'))

    # Pier: 2 rows of pilings from the shoreline out to ~3 m depth + deck.
    sx, sy, si = nearest_where(np.abs(Z) < 0.2, X, Y, (40.0, 160.0))
    down = np.array([-gx[si], gy[si]])
    down /= np.linalg.norm(down)
    perp = np.array([-down[1], down[0]])
    top, spacing, rows = 1.2, 4.0, 6
    for k in range(rows):
        for side in (-1, 1):
            x, y = np.array([sx, sy]) + down * (2.0 + spacing * k) + perp * side * 1.5
            zb = bed(x, y)
            n = f'gt_piling_{k}_{"l" if side < 0 else "r"}'
            snippets.append(cyl_model(n, add('cylinder', n, float(x), float(y),
                                             (zb + top) / 2, radius=0.15,
                                             length=top - zb, bed_z=zb),
                                      0.15, top - zb, '0.35 0.25 0.15 1'))
    length = spacing * (rows - 1) + 4.0
    cx, cy = np.array([sx, sy]) + down * (2.0 + spacing * (rows - 1) / 2)
    yaw = math.atan2(down[1], down[0])
    snippets.append(box_model('gt_pier_deck',
                              add('box', 'gt_pier_deck', float(cx), float(cy),
                                  top + 0.15, yaw, size=[length, 3.6, 0.3]),
                              (length, 3.6, 0.3), '0.55 0.45 0.35 1'))

    # Shore landmarks (LiDAR georeferencing checks), 1 x 1 x 2 m.
    free = np.hypot(gx, gy) < 0.35
    for i, seed in enumerate([(-60.0, 175.0), (215.0, -60.0), (-20.0, -200.0),
                              (-250.0, 110.0)]):
        x, y, _ = nearest_where((Z > 1.0) & (Z < 3.0) & free, X, Y, seed)
        free &= np.hypot(X - x, Y - y) > 60.0
        zb = bed(x, y)
        n = f'gt_landmark_{i}'
        snippets.append(box_model(n, add('box', n, x, y, zb + 1.0, 0.0,
                                         size=[1, 1, 2], bed_z=zb),
                                  (1, 1, 2), '0.95 0.1 0.1 1'))

    waves = tl.wave_params('calm', 0.0)
    includes = [tl.include(name, 'terrain')] + snippets
    lat, lon, elev = CLAYTOR_LAKE
    tl.write_world(os.path.join(WORLDS, 'claytor_lake_calm.sdf'),
                   'claytor_lake_calm', lat, lon, elev, includes, waves,
                   camera_pose='-60 -260 90 0 0.35 1.3',
                   comment='Still-water calibration world (Milestone 1).')

    meta = {'world': 'claytor_lake_calm', 'targets': targets, **extra,
            'pose_note': 'pose z is the primitive centre; yaw about +z',
            'sea_state': 'calm'}
    tl.save_ground_truth(GT, 'claytor_lake_calm', Z, g['dx'], g['x_west'],
                         g['y_north'], lat, lon, meta, masks)
    tl.save_preview(os.path.join(GT, 'claytor_lake_calm.png'), Z, g['dx'],
                    g['x_west'], g['y_north'], 'claytor_lake_calm terrain',
                    vlim=(-16, 50))


# ===========================================================================
# 2. NBS surf zone, repeat epochs
# ===========================================================================

NBS_EPOCHS = {
    0: dict(label='Survey 1 (Oct 2026) baseline',
            retreat_B=0.0, retreat_C=0.0, retreat_A=0.0,
            dune_loss=0.0, dune_toe_retreat=0.0,
            bar_shift=0.0, bar_gain=1.0, berm_z=1.2,
            lee_dep=0.0, gap_scour=0.0, salient_growth=0.0,
            sill_damage=0.0),
    1: dict(label='Survey 2 (Jan 2027) post-storm',
            retreat_B=6.0, retreat_C=2.0, retreat_A=1.5,
            dune_loss=0.8, dune_toe_retreat=5.0,
            bar_shift=20.0, bar_gain=0.8, berm_z=0.9,
            lee_dep=0.15, gap_scour=0.35, salient_growth=3.0,
            sill_damage=0.0),
    2: dict(label='Survey 3 (Mar 2027) partial recovery, sill damage',
            retreat_B=3.5, retreat_C=1.0, retreat_A=1.8,
            dune_loss=0.8, dune_toe_retreat=5.0,
            bar_shift=12.0, bar_gain=0.9, berm_z=1.05,
            lee_dep=0.20, gap_scour=0.45, salient_growth=4.0,
            sill_damage=0.35),
}

NBS_GRID = dict(dx=1.0, x_west=-320.0, x_east=320.0, y_south=-180.0,
                y_north=320.0)
SILLS = [(-290.0 + 45.0 * k, -290.0 + 45.0 * k + 35.0) for k in range(5)]
SILL_D = 38.0          # sill centreline offshore of marsh edge [m]
BREAKWATERS = [120.0, 200.0, 280.0]
BW_D, BW_LEN = 65.0, 45.0
DAMAGED_SILL = 2


def build_nbs(ep):
    g = NBS_GRID
    dx = g['dx']
    nx = int(round((g['x_east'] - g['x_west']) / dx)) + 1
    ny = int(round((g['y_north'] - g['y_south']) / dx)) + 1
    x = g['x_west'] + dx * np.arange(nx)
    X, Y = np.meshgrid(x, g['y_north'] - dx * np.arange(ny))

    # Section weights (smooth transitions over ~30 m).
    wA = 1.0 - smoothstep(-100, -70, X)
    wC = smoothstep(50, 80, X)
    wB = 1.0 - wA - wC
    wA1, wB1, wC1 = wA[0], wB[0], wC[0]

    # ---- shoreline -------------------------------------------------------
    # D0: cross-shore coordinate w.r.t. the fixed baseline shoreline (+ = sea).
    # Structures, dune and upland are anchored to D0; the active beach and
    # shoreface are anchored to D, which shifts landward by `retreat` and
    # tapers to zero change at the depth of closure (~D0 = 260 m).
    ys0 = 3.0 * np.sin(2 * np.pi * x / 260.0) + 2.0 * np.sin(2 * np.pi * x / 97.0)
    salient = sum(np.exp(-((x - xc) / 28.0) ** 2) for xc in BREAKWATERS)
    ys_base = ys0 + wC1 * 14.0 * salient
    retreat = (wA1 * ep['retreat_A'] + wB1 * ep['retreat_B']
               + wC1 * ep['retreat_C'] * (1 - salient))
    growth = wC1 * ep['salient_growth'] * salient      # lee of breakwaters only
    D0 = Y - ys_base[None, :]
    D = (D0 + retreat[None, :] * (1.0 - smoothstep(120, 260, D0))
         - growth[None, :] * (1.0 - smoothstep(30, 60, D0)))

    # ---- beach / dune profile (sections B, C) ---------------------------
    berm = ep['berm_z']
    fore = np.minimum(berm, -D / 10.0)
    toe = -38.0 - ep['dune_toe_retreat'] * wB
    crest = 4.0 - ep['dune_loss'] * wB * smoothstep(-75, -48, D0)
    dune = berm + (crest - berm) * smoothstep(0, 1, (toe - D0) / 10.0)
    back = 4.0 - 1.5 * smoothstep(-70, -110, D0)
    land_beach = np.where(D0 > toe, fore, np.where(D0 > -70, dune, back))

    sea = np.maximum(-0.12 * np.maximum(D, 0.0) ** (2.0 / 3.0), -9.0)  # Dean
    db = 75.0 + ep['bar_shift'] + 12.0 * np.sin(2 * np.pi * X / 180.0)
    hb, wb = 0.7 * ep['bar_gain'], 12.0
    bar = hb * (np.exp(-((D - db) / wb) ** 2)
                - 0.45 * np.exp(-((D - db + 1.4 * wb) / wb) ** 2))
    sea = sea + bar * smoothstep(20, 45, D)
    z_beach = np.where(D < 0, land_beach, sea)

    # ---- living shoreline (section A) ------------------------------------
    marsh_z = 0.35 + fixed_noise(X.shape, 21, 8, 0.04)
    creek_c = -205.0 + 14.0 * np.sin(Y / 23.0)
    creek_d = np.abs(X - creek_c)
    creek_on = (D0 < 0) & (Y > -175)
    creek = creek_on & (creek_d < 7)
    marsh = np.where(creek_on, marsh_z - 0.95 * np.exp(-(creek_d / 3.5) ** 2),
                     marsh_z)
    land_A = np.where(D0 > -110, marsh, 0.35 + 1.8 * smoothstep(-110, -150, D0))
    scarp = 0.35 - 0.5 * smoothstep(0, 1.5, D)          # eroding marsh edge
    terrace = -0.15 - 0.012 * D                          # sheltered sand fill
    blendA = smoothstep(45, 70, D)
    sea_A = np.where(D < 1.5, scarp, terrace * (1 - blendA) + sea * blendA)
    z_A = np.where(D < 0, land_A, sea_A)
    z_A += ep['lee_dep'] * np.exp(-((D0 - 22.0) / 9.0) ** 2) * (D0 > 3)

    # Oyster-reef sills: crest +0.30 m, crest width 2 m, side slope 1:2.
    sill_z = np.full(X.shape, -1e9)
    for k, (xa, xb) in enumerate(SILLS):
        damaged = k == DAMAGED_SILL and ep['sill_damage'] > 0
        top = 0.30 - (ep['sill_damage'] if damaged else 0.0)
        half_crest = 2.5 if damaged else 1.0       # displaced rock spreads
        along = np.maximum(np.maximum(xa - X, X - xb), 0.0)
        across = np.maximum(np.abs(D0 - SILL_D) - half_crest, 0.0)
        sill_z = np.maximum(sill_z, top - 0.5 * np.hypot(along, across))
        if k < len(SILLS) - 1 and ep['gap_scour'] > 0:  # scour at gap exits
            gx = (xb + SILLS[k + 1][0]) / 2.0
            z_A -= ep['gap_scour'] * np.exp(
                -(((X - gx) / 4.0) ** 2 + ((D0 - SILL_D - 5.0) / 4.0) ** 2))
    sill_mask = sill_z > z_A + 0.05
    z_A = np.maximum(z_A, sill_z)

    # ---- detached rock breakwaters (section C) --------------------------
    bw_z = np.full(X.shape, -1e9)
    Db = Y - (ys0[None, :] + BW_D)
    for xc in BREAKWATERS:
        along = np.maximum(np.abs(X - xc) - BW_LEN / 2, 0.0)
        across = np.maximum(np.abs(Db) - 1.5, 0.0)
        bw_z = np.maximum(bw_z, 1.5 - np.hypot(along, across) / 1.5)
    bw_mask = bw_z > z_beach + 0.05
    z_C = np.maximum(z_beach, bw_z)

    Z = wA * z_A + wB * z_beach + wC * z_C

    # Shared fine-scale roughness (identical across epochs).
    rock_mask = bw_mask & (wC > 0.5)
    oyster_mask = sill_mask & (wA > 0.5)
    Z = Z + fixed_noise(Z.shape, 7, 1.5, 0.03)
    Z = Z + np.where(rock_mask | oyster_mask,
                     fixed_noise(Z.shape, 8, 0.7, 0.12), 0.0)

    masks = {
        'grass': (Z > 2.6) & (wA < 0.5),
        'marsh': (wA > 0.5) & (D < -0.5) & (Z > 0.1) & ~creek,
        'creek': (wA > 0.5) & creek & (Z < 0.1),
        'oyster': oyster_mask,
        'rock': rock_mask,
    }
    return Z.astype(np.float32), masks, ys_base - retreat + growth


def shoreline_positions(Z, g, masks):
    """Cross-shore position of the most seaward z=0 crossing of the natural
    shore (sill / breakwater crests excluded) for each x column."""
    Z = np.where(masks['oyster'] | masks['rock'], -1.0, Z)
    ny, nx = Z.shape
    y = g['y_north'] - g['dx'] * np.arange(ny)
    out = np.full(nx, np.nan)
    for j in range(nx):
        col = Z[:, j]
        idx = np.where((col[:-1] < 0) & (col[1:] >= 0))[0]  # north->south
        if idx.size:
            i = idx[0]
            t = -col[i] / (col[i + 1] - col[i])
            out[j] = y[i] - t * g['dx']
    return out


def generate_nbs(args):
    g = NBS_GRID
    lat, lon, elev = HAMPTON_ROADS
    waves = tl.wave_params(args.sea_state, -math.pi / 2)   # travelling south
    Zs, shore = {}, {}
    for e, ep in NBS_EPOCHS.items():
        world = f'nbs_surf_epoch{e}'
        print(f'[{world}] {ep["label"]}')
        Z, masks, _ = build_nbs(ep)
        Zs[e] = Z
        model = f'nbs_surf_terrain_epoch{e}'
        tl.write_terrain_model(MODELS, model, Z, g['dx'], g['x_west'],
                               g['y_north'], masks,
                               description=f'NBS surf-zone terrain, {ep["label"]}',
                               seed=3)
        tl.write_world(os.path.join(WORLDS, f'{world}.sdf'), world, lat, lon,
                       elev, [tl.include(model, 'terrain')], waves,
                       wind_speed=0.0,
                       camera_pose='-40 380 90 0 0.45 -1.57',
                       comment=f'{ep["label"]}. Sections: A living shoreline '
                               '(x<-85), B control beach, C breakwaters (x>65).')
        meta = {'world': world, 'epoch': e, 'epoch_params': ep,
                'sea_state': args.sea_state,
                'sections_x_m': {'A_living_shoreline': [-320, -85],
                                 'B_control_beach': [-85, 65],
                                 'C_breakwaters': [65, 320]},
                'sills_x_extent_m': SILLS, 'sill_offshore_m': SILL_D,
                'damaged_sill_index': DAMAGED_SILL,
                'breakwater_centres_x_m': BREAKWATERS}
        tl.save_ground_truth(GT, world, Z, g['dx'], g['x_west'], g['y_north'],
                             lat, lon, meta, masks)
        tl.save_preview(os.path.join(GT, f'{world}.png'), Z, g['dx'],
                        g['x_west'], g['y_north'], f'{world}: {ep["label"]}',
                        vlim=(-8, 4.5))
        shore[e] = shoreline_positions(Z, g, masks)

    # Change ground truth
    summary = {}
    cell_area = g['dx'] ** 2
    xs = g['x_west'] + g['dx'] * np.arange(Zs[0].shape[1])
    sect = {'A_living_shoreline': xs < -85, 'B_control_beach': (xs >= -85) & (xs < 65),
            'C_breakwaters': xs >= 65}
    for a, b in [(0, 1), (1, 2), (0, 2)]:
        dz = Zs[b] - Zs[a]
        key = f'change_epoch{a}_to_epoch{b}'
        np.savez_compressed(os.path.join(GT, key + '.npz'), dz=dz,
                            shoreline_y_a=shore[a], shoreline_y_b=shore[b], x=xs)
        tl.save_preview(os.path.join(GT, key + '.png'), dz, g['dx'],
                        g['x_west'], g['y_north'], key, diff=True, vlim=1.0)
        s = {}
        for name, cm in sect.items():
            d = dz[:, cm]
            s[name] = {
                'erosion_m3': float(-d[d < -0.02].sum() * cell_area),
                'deposition_m3': float(d[d > 0.02].sum() * cell_area),
                'mean_shoreline_change_m': float(np.nanmean(shore[b][cm] - shore[a][cm])),
            }
        summary[key] = s
    with open(os.path.join(GT, 'nbs_change_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=1))

    # Repeat-survey transects (identical in every epoch).
    lines = []
    for i, xl in enumerate(np.arange(-300, 301, 25.0)):
        lines.append({'id': i, 'start_xy': [float(xl), 260.0],
                      'end_xy': [float(xl), 12.0]})
    with open(os.path.join(CONFIG, 'nbs_survey_transects.yaml'), 'w') as f:
        f.write('# Cross-shore survey lines (world ENU, m). Offshore -> onshore.\n'
                '# Inshore ends stop ~10 m off the baseline shoreline; adjust\n'
                '# for vessel draft / sea state.\n')
        f.write('frame: world_enu\nspacing_m: 25.0\nlines:\n')
        for ln in lines:
            f.write(f"  - {{id: {ln['id']}, start: [{ln['start_xy'][0]}, "
                    f"{ln['start_xy'][1]}], end: [{ln['end_xy'][0]}, "
                    f"{ln['end_xy'][1]}]}}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sea-state', default='moderate', choices=tl.SEA_STATES,
                   help='sea state for the NBS surf-zone worlds')
    p.add_argument('--only', choices=['lake', 'nbs'])
    args = p.parse_args()
    for d in (MODELS, WORLDS, GT, CONFIG):
        os.makedirs(d, exist_ok=True)
    if args.only in (None, 'lake'):
        generate_lake(args)
    if args.only in (None, 'nbs'):
        generate_nbs(args)


if __name__ == '__main__':
    main()
