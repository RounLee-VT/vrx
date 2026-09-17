#!/usr/bin/env python3
"""Surface material maps for LiDAR intensity / sonar backscatter synthesis.

Builds ground_truth/<world>_materials.npz (+ a preview PNG) for every coastal
world from the exported ground truth (DEMs and class masks) and elevation
rules, without touching the terrain meshes. Material ids and their LiDAR
reflectivity / sonar backscatter are defined in
vrx_gz/config/coastal_materials.yaml and read at runtime by
vrx_gz/coastal_materials.py.

Rules (z relative to still water):
  z < -0.3            submerged: sand_seabed (mud in the reservoir and in
                      tidal creek beds)
  -0.3 <= z < 0.6     swash / splash zone: sand_wet, rock_wet
  0.6 <= z < 3        sand_dry (beaches, berms)
  z >= 3              grass (dunes, uplands)
  plus the explicit masks of each world (rock, oyster reef, marsh, creek,
  talus, algae, forest, concrete ramp, ...), and for the real Ocean View DEM
  steep near-shore cells (riprap breakwaters / jetties) as rock.

Usage: python3 gen_material_maps.py [--worlds ...]
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_synthetic import GT, VRX_GZ, fixed_noise  # noqa: E402

TABLE = os.path.join(VRX_GZ, 'config', 'coastal_materials.yaml')
with open(TABLE) as f:
    MATS = {m['name']: m['id'] for m in yaml.safe_load(f)['materials']}
PALETTE = {
    'unknown': (0.5, 0.5, 0.5), 'sand_dry': (0.93, 0.85, 0.60),
    'sand_wet': (0.72, 0.62, 0.42), 'sand_seabed': (0.55, 0.60, 0.70),
    'mud': (0.30, 0.25, 0.20), 'grass': (0.40, 0.65, 0.30),
    'forest': (0.15, 0.40, 0.15), 'marsh': (0.55, 0.65, 0.30),
    'bare_soil': (0.60, 0.48, 0.35), 'rock': (0.60, 0.58, 0.55),
    'rock_wet': (0.38, 0.37, 0.36), 'algae': (0.20, 0.30, 0.15),
    'oyster_reef': (0.80, 0.75, 0.65), 'talus': (0.85, 0.65, 0.45),
    'concrete': (0.85, 0.85, 0.85), 'object': (0.9, 0.2, 0.2)}


def by_elevation(z):
    cls = np.full(z.shape, MATS['grass'], np.uint8)
    cls[z < 3.0] = MATS['sand_dry']
    cls[z < 0.6] = MATS['sand_wet']
    cls[z < -0.3] = MATS['sand_seabed']
    return cls


def wet_rock(cls, z, mask):
    """Rock, wet in the splash / swash band (-0.3 <= z < 0.6)."""
    cls[mask] = MATS['rock']
    cls[mask & (z < 0.6) & (z >= -0.3)] = MATS['rock_wet']


def load(name):
    return np.load(os.path.join(GT, name))


def save(world, kind, outside, preview, **arrays):
    info = json.load(open(os.path.join(GT, f'{world}.json')))
    g = info.get('ground_grid', info)
    np.savez_compressed(os.path.join(GT, f'{world}_materials.npz'), kind=kind,
                        outside_class=outside, dx=g['dx_m'], x_west=g['x_west_m'],
                        y_north=g['y_north_m'], **arrays)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    names = {v: k for k, v in MATS.items()}
    rgb = np.zeros(preview.shape + (3,))
    for i in np.unique(preview):
        rgb[preview == i] = PALETTE[names[int(i)]]
    ny, nx = preview.shape
    ext = [g['x_west_m'], g['x_west_m'] + g['dx_m'] * (nx - 1),
           g['y_north_m'] - g['dx_m'] * (ny - 1), g['y_north_m']]
    fig, ax = plt.subplots(figsize=(11, 11 * ny / nx + 1))
    ax.imshow(rgb, extent=ext)
    ax.legend(handles=[Patch(color=PALETTE[names[int(i)]], label=names[int(i)])
                       for i in np.unique(preview)], loc='upper right', fontsize=8)
    ax.set_title(f'{world}: surface materials')
    ax.set_xlabel('x east [m]')
    ax.set_ylabel('y north [m]')
    fig.tight_layout()
    fig.savefig(os.path.join(GT, f'{world}_materials.png'), dpi=70)
    plt.close(fig)
    counts = {names[int(i)]: int((preview == i).sum()) for i in np.unique(preview)}
    print(f'  {world}: {counts}')


def ocean_view(world='ocean_view_norfolk'):
    z = load(f'{world}.npz')['z'].astype(float)
    cls = by_elevation(z)
    info = json.load(open(os.path.join(GT, f'{world}.json')))
    gy, gx = np.gradient(z, info['dx_m'])
    riprap = (np.hypot(gx, gy) > 0.35) & (z > -3.0) & (z < 4.0)
    wet_rock(cls, z, riprap)
    save(world, 'dem', MATS['sand_seabed'], cls, z=z.astype(np.float32), cls=cls)


def nbs(world):
    d = load(f'{world}.npz')
    z = d['z'].astype(float)
    cls = by_elevation(z)
    cls[d['mask_grass']] = MATS['grass']
    cls[d['mask_marsh']] = MATS['marsh']
    cls[d['mask_creek']] = MATS['mud']
    cls[d['mask_oyster']] = MATS['oyster_reef']
    wet_rock(cls, z, d['mask_rock'])
    save(world, 'dem', MATS['sand_seabed'], cls, z=z.astype(np.float32), cls=cls)


def lake(world='claytor_lake_calm'):
    d = load(f'{world}.npz')
    z = d['z'].astype(float)
    cls = np.full(z.shape, MATS['grass'], np.uint8)
    cls[z < 0.5] = MATS['bare_soil']          # draw-down shoreline
    cls[z < -1.5] = MATS['mud']               # reservoir silt
    cls[d['mask_grass']] = MATS['grass']
    cls[d['mask_forest']] = MATS['forest']
    wet_rock(cls, z, d['mask_rock'])
    cls[d['mask_concrete']] = MATS['concrete']
    save(world, 'dem', MATS['forest'], cls, z=z.astype(np.float32), cls=cls)


def cliff(world):
    import gen_cliff_coast as gc
    e = int(world[-1])
    gd = load(f'{world}_ground.npz')
    pd = load(f'{world}_plateau.npz')
    gz = gd['z'].astype(float)
    gcls = by_elevation(gz)
    wet_rock(gcls, gz, gd['mask_rock'])
    gcls[gd['mask_algae']] = MATS['algae']
    sand = gd['mask_sand']
    gcls[sand & (gz >= 0.6)] = MATS['sand_dry']
    gcls[sand & (gz < 0.6)] = MATS['sand_wet']
    gcls[sand & (gz < -0.3)] = MATS['sand_seabed']
    gcls[gd['mask_talus']] = MATS['talus']

    # plateau classes from the generator's own masks (grass / bare path /
    # shrubs), identical to the texture
    _, _, _, face = gc.build_face(gc.EPOCHS[e])
    _, _, Zp, _, _, pmasks = gc.build_plateau(face)
    pcls = np.full(Zp.shape, MATS['grass'], np.uint8)
    pcls[fixed_noise(Zp.shape, 93, 1.2, 1.0) > 1.3] = MATS['forest']
    pcls[pmasks['bare']] = MATS['bare_soil']

    preview = np.where(pd['mask_valid'], pcls, np.where(gd['mask_valid'], gcls, MATS['rock']))
    save(world, 'cliff', MATS['sand_seabed'], preview,
         ground_z=gz.astype(np.float32), ground_cls=gcls, ground_valid=gd['mask_valid'],
         plateau_z=pd['z'].astype(np.float32), plateau_cls=pcls,
         plateau_valid=pd['mask_valid'])


WORLDS = {
    'claytor_lake_calm': lake,
    'ocean_view_norfolk': ocean_view,
    **{f'nbs_surf_epoch{e}': nbs for e in range(3)},
    **{f'cliff_coast_epoch{e}': cliff for e in range(3)},
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--worlds', nargs='*', default=list(WORLDS))
    for w in p.parse_args().worlds:
        fn = WORLDS[w]
        if fn in (lake, ocean_view):
            fn()
        else:
            fn(w)


if __name__ == '__main__':
    main()
