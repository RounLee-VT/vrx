#!/usr/bin/env python3
"""Real topobathy world: East Ocean View / Little Creek Inlet, Norfolk VA.

Source: NOAA NCEI CUDEM 1/9 arc-second (~3 m) topobathymetric DEM,
tile ncei19_n37x00_w076x25_2019v1 (horizontal NAD83, vertical NAVD88, metres),
read in-place from the NOAA Digital Coast public S3 bucket (only the needed
window is fetched). The bay shoreline here has a sandy surf zone, a series of
nearshore breakwaters with crenulate pocket beaches, dunes, and the Little
Creek Inlet jetties - a realistic Chesapeake Bay / Hampton Roads test site.

The DEM is reprojected onto a local transverse-Mercator grid centred on the
world origin (== Gazebo spherical-coordinates ENU frame) and shifted so that
the requested water level (NAVD88) is z = 0 in simulation.

Requires: rasterio (pip install rasterio), numpy, scipy, pillow, matplotlib.

Usage:
  python3 gen_ocean_view.py [--water-level 0.0] [--sea-state moderate]
                            [--res 3.0] [--bounds W S E N]
"""

import argparse
import math
import os

import numpy as np
from scipy import ndimage

import terrain_lib as tl

# scripts live in vrx_gz/worlds/coastal/scripts
VRX_GZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
TILE_URL = ('https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/'
            'NCEI_ninth_Topobathy_2014_8483/chesapeake_bay/'
            'ncei19_n37x00_w076x25_2019v1.tif')
DEFAULT_BOUNDS = (-76.1985, 36.9255, -76.1725, 36.9405)   # W S E N (deg)


def fetch_dem(bounds, res, cache):
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import from_bounds

    w, s, e, n = bounds
    lat0, lon0 = (s + n) / 2.0, (w + e) / 2.0
    dst_crs = CRS.from_proj4(tl.local_tm_proj(lat0, lon0))

    # Local metric extent (inscribed in the lon/lat box).
    half_x = (e - w) / 2.0 * 111320.0 * math.cos(math.radians(lat0)) - 2 * res
    half_y = (n - s) / 2.0 * 110950.0 - 2 * res
    half_x = math.floor(half_x / res) * res
    half_y = math.floor(half_y / res) * res
    nx = int(round(2 * half_x / res)) + 1
    ny = int(round(2 * half_y / res)) + 1
    x_west, y_north = -half_x, half_y

    if cache and os.path.exists(cache):
        src_path = cache
    else:
        src_path = '/vsicurl/' + TILE_URL
    print(f'reading {src_path}')
    with rasterio.open(src_path) as src:
        pad = 0.002
        win = from_bounds(w - pad, s - pad, e + pad, n + pad, src.transform)
        win = win.round_offsets().round_lengths()
        data = src.read(1, window=win).astype(np.float32)
        src_tr = src.window_transform(win)
        nodata = src.nodata
        src_crs = src.crs
    if nodata is not None:
        bad = data == nodata
        if bad.any():
            idx = ndimage.distance_transform_edt(bad, return_distances=False,
                                                 return_indices=True)
            data = data[tuple(idx)]

    Z = np.zeros((ny, nx), np.float32)
    # Vertex-centred grid: pixel centre (col, row) <-> vertex (col, row).
    dst_tr = from_origin(x_west - res / 2, y_north + res / 2, res, res)
    reproject(data, Z, src_transform=src_tr, src_crs=src_crs,
              dst_transform=dst_tr, dst_crs=dst_crs,
              resampling=Resampling.bilinear)
    return Z, x_west, y_north, lat0, lon0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bounds', nargs=4, type=float, default=DEFAULT_BOUNDS,
                   metavar=('W', 'S', 'E', 'N'))
    p.add_argument('--res', type=float, default=3.0, help='grid spacing [m]')
    p.add_argument('--water-level', type=float, default=0.0,
                   help='still-water level in NAVD88 metres (tide stage)')
    p.add_argument('--sea-state', default='moderate', choices=tl.SEA_STATES)
    p.add_argument('--wave-dir-deg', type=float, default=-100.0,
                   help='wave propagation direction, ENU deg (-90 = to south)')
    p.add_argument('--name', default='ocean_view_norfolk')
    p.add_argument('--cache', help='optional local copy of the CUDEM tile')
    args = p.parse_args()

    Z, x_west, y_north, lat0, lon0 = fetch_dem(args.bounds, args.res, args.cache)
    Z = Z - args.water_level
    print(f'grid {Z.shape[1]} x {Z.shape[0]} @ {args.res} m, '
          f'z [{Z.min():.1f}, {Z.max():.1f}] m, origin {lat0:.6f}, {lon0:.6f}')

    model = f'{args.name}_terrain'
    masks = {'grass': ndimage.gaussian_filter(Z, 1) > 4.5}
    tl.write_terrain_model(os.path.join(VRX_GZ, 'models'), model, Z, args.res,
                           x_west, y_north, masks,
                           description='NOAA CUDEM topobathy, East Ocean View / '
                                       'Little Creek Inlet, Norfolk VA', seed=5)

    waves = tl.wave_params(args.sea_state, math.radians(args.wave_dir_deg))
    tl.write_world(os.path.join(VRX_GZ, 'worlds', f'{args.name}.sdf'), args.name,
                   lat0, lon0, 0.0, [tl.include(model, 'terrain')], waves,
                   camera_pose=f'0 {y_north + 150:.0f} 160 0 0.5 -1.57',
                   comment='Real topobathy: NOAA NCEI CUDEM 1/9 arc-sec, '
                           f'water level {args.water_level:+.2f} m NAVD88.')

    gt = os.path.join(VRX_GZ, 'worlds', 'coastal', 'ground_truth')
    meta = {'world': args.name, 'source': TILE_URL,
            'source_citation': 'NOAA NCEI Continuously Updated Digital '
                               'Elevation Model (CUDEM) - 1/9 Arc-Second '
                               'Resolution Bathymetric-Topographic Tiles',
            'source_datums': {'horizontal': 'NAD83', 'vertical': 'NAVD88'},
            'bounds_lonlat_WSEN': list(args.bounds),
            'water_level_navd88_m': args.water_level,
            'z_to_navd88': f'z_navd88 = z + {args.water_level}',
            'sea_state': args.sea_state}
    tl.save_ground_truth(gt, args.name, Z, args.res, x_west, y_north, lat0,
                         lon0, meta)
    tl.save_preview(os.path.join(gt, f'{args.name}.png'), Z, args.res, x_west,
                    y_north, f'{args.name} (CUDEM, WL {args.water_level:+.2f} m '
                    'NAVD88)', vlim=(-9, 6))


if __name__ == '__main__':
    main()
