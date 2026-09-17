"""Shared utilities: heightfield -> Gazebo model, world SDF, ground-truth export.

Conventions (all generators):
  * World frame is ENU: +x east, +y north, +z up, metres.
  * A heightfield ``Z`` has shape (ny, nx); row 0 is the NORTH edge, column 0
    the WEST edge. Grid spacing ``dx`` is equal in x and y.
  * ``z = 0`` is the still-water level used by the VRX wave/buoyancy plugins
    (``fluid_level`` = 0). Real DEMs are shifted by the chosen water level.
"""

import json
import math
import os

import numpy as np
from PIL import Image
from scipy import ndimage

# ---------------------------------------------------------------------------
# Sea-state presets (VRX PMS wavefield: 3 components, scale 1.1)
# ---------------------------------------------------------------------------

G = 9.81


def _pms_hs(period, gain=1.0, scale=1.1, number=3):
    """Significant wave height that vrx::Wavefield (PMS model) produces."""
    wp = 2.0 * math.pi / period
    spacing = [wp * (1.0 - 1.0 / scale),
               wp * (scale - 1.0 / scale) / 2.0,
               wp * (scale - 1.0)]
    var = 0.0
    for i in range(number):
        w = wp * scale ** (i - 1)
        s = 0.0081 * G ** 2 / w ** 5 * math.exp(-1.25 * (wp / w) ** 4)
        a = gain * math.sqrt(2.0 * s * spacing[i])
        var += a * a / 2.0
    return 4.0 * math.sqrt(var)


# Hs targets are typical of fetch-limited Chesapeake Bay wind seas.
SEA_STATES = {
    'calm':     {'hs': 0.0, 'period': 3.0},
    'light':    {'hs': 0.25, 'period': 2.5},
    'moderate': {'hs': 0.5, 'period': 3.5},
    'rough':    {'hs': 1.0, 'period': 4.5},
}


def wave_params(sea_state, direction_rad):
    p = SEA_STATES[sea_state]
    gain = 0.0 if p['hs'] == 0 else p['hs'] / _pms_hs(p['period'])
    return {'gain': gain, 'period': p['period'], 'direction': direction_rad,
            'steepness': 0.0, 'hs': p['hs']}


# ---------------------------------------------------------------------------
# Texture
# ---------------------------------------------------------------------------

def _lerp(c0, c1, t):
    t = np.clip(t, 0.0, 1.0)[..., None]
    return np.asarray(c0) * (1 - t) + np.asarray(c1) * t


def hillshade(Z, dx, azimuth=315.0, altitude=45.0):
    gy, gx = np.gradient(Z, dx)
    gy = -gy  # row index grows southward
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)
    shade = (np.sin(alt) * np.sin(slope)
             + np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    return np.clip(shade, 0, 1)


def colorize(Z, dx, masks=None, seed=0):
    """Elevation-based albedo with optional class masks (bool arrays)."""
    masks = masks or {}
    rng = np.random.default_rng(seed)
    noise = ndimage.gaussian_filter(rng.standard_normal(Z.shape), 1.0)
    noise = noise / (np.abs(noise).max() + 1e-9)

    rgb = _lerp((0.70, 0.64, 0.46), (0.30, 0.30, 0.24), -Z / 10.0)  # seabed
    wet = (Z > -0.3) & (Z <= 0.5)
    rgb[wet] = (0.60, 0.53, 0.38)
    dry = (Z > 0.5) & (Z <= 2.5)
    rgb[dry] = (0.86, 0.80, 0.62)
    high = Z > 2.5
    rgb[high] = _lerp((0.80, 0.75, 0.56), (0.42, 0.50, 0.30),
                      (Z[high] - 2.5) / 1.0)

    palette = {
        'grass': (0.36, 0.46, 0.26),
        'marsh': (0.44, 0.52, 0.27),
        'creek': (0.35, 0.32, 0.25),
        'rock': (0.47, 0.45, 0.42),
        'oyster': (0.55, 0.51, 0.43),
        'concrete': (0.70, 0.70, 0.68),
        'forest': (0.20, 0.30, 0.16),
        'bare': (0.60, 0.54, 0.40),
        'algae': (0.26, 0.29, 0.19),
        'talus': (0.66, 0.58, 0.44),
        'sand': (0.82, 0.74, 0.55),
    }
    for name, m in masks.items():
        if m is not None and m.any():
            rgb[m] = palette[name]

    rough = np.zeros(Z.shape, bool)
    for name in ('rock', 'oyster', 'talus', 'algae'):
        if masks.get(name) is not None:
            rough |= masks[name]
    amp = np.where(rough, 0.18, 0.05)
    rgb = rgb * (1.0 + amp[..., None] * noise[..., None])
    rgb = rgb * (0.65 + 0.35 * hillshade(Z, dx))[..., None]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Mesh writing (Wavefront OBJ)
# ---------------------------------------------------------------------------

def _grid_xy(shape, dx, x_west, y_north):
    ny, nx = shape
    x = x_west + dx * np.arange(nx)
    y = y_north - dx * np.arange(ny)
    return np.meshgrid(x, y)


def _cell_faces(ny, nx, rows, cols):
    """Two CCW (seen from +z) triangles per cell (r, c); 0-based vertex ids."""
    v00 = rows * nx + cols
    v01 = v00 + 1
    v10 = v00 + nx
    v11 = v10 + 1
    t1 = np.stack([v00, v10, v11], axis=-1)
    t2 = np.stack([v00, v11, v01], axis=-1)
    return np.concatenate([t1.reshape(-1, 3), t2.reshape(-1, 3)])


def write_obj(path, Z, dx, x_west, y_north, texture=None, chunk=192,
              cell_mask=None):
    """Write a heightfield as OBJ with one self-contained sub-mesh per chunk.

    Every sub-mesh carries its own v/vt/vn block (border vertices duplicated):
    Gazebo's OBJ loader otherwise copies the whole vertex list into each
    sub-mesh, and dartsim rejects sub-meshes without per-vertex normals.

    texture:   albedo image file name (UVs span the full heightfield).
    chunk:     cells per side of each sub-mesh (renderer culling).
    cell_mask: (ny-1, nx-1) bool; only True cells are emitted.
    """
    ny, nx = Z.shape
    X, Y = _grid_xy(Z.shape, dx, x_west, y_north)
    gy, gx = np.gradient(Z, dx)
    N = np.stack([-gx, gy, np.ones_like(Z)], -1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    U = (np.arange(nx) / (nx - 1))[None, :].repeat(ny, 0)
    V = (1.0 - np.arange(ny) / (ny - 1))[:, None].repeat(nx, 1)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    offset, n_verts, n_tris = 0, 0, 0
    with open(path, 'w') as f:
        f.write('# vrx_gz coastal heightfield mesh\n')
        f.write(f'mtllib {stem}.mtl\n')
        for r0 in range(0, ny - 1, chunk):
            for c0 in range(0, nx - 1, chunk):
                r1, c1 = min(r0 + chunk, ny - 1), min(c0 + chunk, nx - 1)
                rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1),
                                     indexing='ij')
                if cell_mask is not None:
                    keep = cell_mask[rr, cc]
                    rr, cc = rr[keep], cc[keep]
                if rr.size == 0:
                    continue
                faces = _cell_faces(ny, nx, rr.ravel(), cc.ravel())
                used = np.unique(faces)
                local = np.searchsorted(used, faces)
                sl = np.unravel_index(used, (ny, nx))
                f.write(f'o chunk_{r0}_{c0}\nusemtl terrain\n')
                np.savetxt(f, np.stack([X[sl], Y[sl], Z[sl]], 1),
                           fmt='v %.3f %.3f %.3f')
                np.savetxt(f, np.stack([U[sl], V[sl]], 1), fmt='vt %.5f %.5f')
                np.savetxt(f, N[sl], fmt='vn %.4f %.4f %.4f')
                idx = np.repeat(local + offset + 1, 3, axis=1)
                np.savetxt(f, idx, fmt='f %d/%d/%d %d/%d/%d %d/%d/%d')
                offset += len(used)
                n_verts += len(used)
                n_tris += len(faces)
    with open(os.path.join(os.path.dirname(path), f'{stem}.mtl'), 'w') as f:
        f.write('newmtl terrain\nKa 1 1 1\nKd 1 1 1\nKs 0.02 0.02 0.02\n'
                'Ns 5\nd 1\nillum 1\n')
        if texture:
            f.write(f'map_Kd {texture}\n')
    return n_verts, n_tris


def grid_part(P, N, UV=None, cell_mask=None):
    """Triangulate a parametric vertex grid.

    P, N: (nr, nc, 3) positions / outward normals; UV: (nr, nc, 2) or None.
    cell_mask: (nr-1, nc-1) bool. Each triangle's winding is chosen so its
    face normal agrees with the supplied vertex normals.
    """
    nr, nc, _ = P.shape
    rr, cc = np.meshgrid(np.arange(nr - 1), np.arange(nc - 1), indexing='ij')
    if cell_mask is not None:
        rr, cc = rr[cell_mask], cc[cell_mask]
    f = _cell_faces(nr, nc, rr.ravel(), cc.ravel())
    return mesh_part(P.reshape(-1, 3), N.reshape(-1, 3),
                     None if UV is None else UV.reshape(-1, 2), f)


def mesh_part(V, N, UV, F):
    """Compact an indexed triangle set and fix per-triangle winding."""
    V, N, F = np.asarray(V, float), np.asarray(N, float), np.asarray(F)
    if len(F) == 0:
        return None
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(b - a, c - a)
    area = np.linalg.norm(fn, axis=1)
    F = F[area > 1e-6]
    fn = fn[area > 1e-6]
    if len(F) == 0:
        return None
    flip = np.einsum('ij,ij->i', fn, N[F].sum(axis=1)) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    used = np.unique(F)
    remap = np.searchsorted(used, F)
    n = N[used] / (np.linalg.norm(N[used], axis=1, keepdims=True) + 1e-12)
    return {'v': V[used], 'n': n,
            'uv': None if UV is None else np.asarray(UV)[used], 'f': remap}


def write_parts_obj(path, parts, texture=None, chunk_tris=150000):
    """Write mesh parts (dicts from mesh_part) as one OBJ with one self-
    contained sub-mesh per part (large parts are split)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    off, nv, nt, k = 0, 0, 0, 0
    with open(path, 'w') as fo:
        fo.write('# vrx_gz coastal mesh\n')
        fo.write(f'mtllib {stem}.mtl\n')
        for part in parts:
            if part is None:
                continue
            for i0 in range(0, len(part['f']), chunk_tris):
                f = part['f'][i0:i0 + chunk_tris]
                used = np.unique(f)
                loc = np.searchsorted(used, f)
                fo.write(f'o part_{k}\nusemtl mat\n')
                k += 1
                np.savetxt(fo, part['v'][used], fmt='v %.3f %.3f %.3f')
                uv = part['uv'][used] if part['uv'] is not None else \
                    np.zeros((len(used), 2))
                np.savetxt(fo, uv, fmt='vt %.5f %.5f')
                np.savetxt(fo, part['n'][used], fmt='vn %.4f %.4f %.4f')
                np.savetxt(fo, np.repeat(loc + off + 1, 3, axis=1),
                           fmt='f %d/%d/%d %d/%d/%d %d/%d/%d')
                off += len(used)
                nv += len(used)
                nt += len(f)
    with open(os.path.join(os.path.dirname(path), f'{stem}.mtl'), 'w') as fo:
        fo.write('newmtl mat\nKa 1 1 1\nKd 1 1 1\nKs 0.02 0.02 0.02\n'
                 'Ns 5\nd 1\nillum 1\n')
        if texture:
            fo.write(f'map_Kd {texture}\n')
    return nv, nt


def heightfield_normals(Z, dx):
    gy, gx = np.gradient(Z, dx)
    N = np.stack([-gx, gy, np.ones_like(Z)], -1)
    return N / np.linalg.norm(N, axis=-1, keepdims=True)


def split_grid(P, N, UV, cell_mask, block=192):
    """grid_part over blocks (renderer culling / smaller sub-meshes)."""
    nr, nc, _ = P.shape
    parts = []
    for r0 in range(0, nr - 1, block):
        for c0 in range(0, nc - 1, block):
            r1, c1 = min(r0 + block, nr - 1), min(c0 + block, nc - 1)
            m = None if cell_mask is None else cell_mask[r0:r1, c0:c1]
            if m is not None and not m.any():
                continue
            parts.append(grid_part(P[r0:r1 + 1, c0:c1 + 1],
                                   N[r0:r1 + 1, c0:c1 + 1],
                                   None if UV is None else UV[r0:r1 + 1, c0:c1 + 1],
                                   m))
    return parts


def write_terrain_model(models_dir, name, Z, dx, x_west, y_north, masks=None,
                        description='', collision_stride=2,
                        collision_zmin=-4.0, texels_per_cell=2, seed=0,
                        surround=None):
    """Create a static Gazebo model (visual + collision mesh + texture).

    surround: optional coarse outer terrain, dict(Z, dx, x_west, y_north,
    cell_mask, masks); rendered as a second visual so the detailed area
    does not end abruptly at its edges.
    """
    root = os.path.join(models_dir, name)
    mdir = os.path.join(root, 'meshes')
    os.makedirs(mdir, exist_ok=True)

    rgb = colorize(Z, dx, masks, seed=seed)
    img = Image.fromarray(rgb)
    ny, nx = Z.shape
    size = (min(8192, nx * texels_per_cell), min(8192, ny * texels_per_cell))
    img.resize(size, Image.BICUBIC).save(os.path.join(mdir, 'terrain.png'))

    nv, nf = write_obj(os.path.join(mdir, 'terrain_visual.obj'), Z, dx,
                       x_west, y_north, texture='terrain.png')

    # Collision: decimated, and only where the hull can actually touch.
    s = max(1, int(collision_stride))
    Zc = Z[::s, ::s]
    cell_max = np.maximum.reduce([Zc[:-1, :-1], Zc[1:, :-1],
                                  Zc[:-1, 1:], Zc[1:, 1:]])
    cmask = cell_max > collision_zmin
    cnv, cnf = write_obj(os.path.join(mdir, 'terrain_collision.obj'), Zc,
                         dx * s, x_west, y_north, cell_mask=cmask)

    extra = ''
    if surround is not None:
        sz = surround
        srgb = colorize(sz['Z'], sz['dx'], sz.get('masks'), seed=seed + 1)
        Image.fromarray(srgb).resize((srgb.shape[1] * 4, srgb.shape[0] * 4),
                                     Image.BICUBIC).save(os.path.join(mdir, 'surround.png'))
        write_obj(os.path.join(mdir, 'surround_visual.obj'), sz['Z'], sz['dx'],
                  sz['x_west'], sz['y_north'], texture='surround.png',
                  cell_mask=sz['cell_mask'])
        extra = """
      <visual name="surround">
        <cast_shadows>false</cast_shadows>
        <geometry><mesh><uri>meshes/surround_visual.obj</uri></mesh></geometry>
      </visual>
      <collision name="surround_collision">
        <geometry><mesh><uri>meshes/surround_visual.obj</uri></mesh></geometry>
      </collision>"""

    with open(os.path.join(root, 'model.config'), 'w') as f:
        f.write(f"""<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author><name>vrx_gz/worlds/coastal/scripts</name></author>
  <description>{description}</description>
</model>
""")
    with open(os.path.join(root, 'model.sdf'), 'w') as f:
        f.write(f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <cast_shadows>false</cast_shadows>
        <geometry>
          <mesh><uri>meshes/terrain_visual.obj</uri></mesh>
        </geometry>
      </visual>
      <collision name="collision">
        <geometry>
          <mesh><uri>meshes/terrain_collision.obj</uri></mesh>
        </geometry>
        <surface>
          <friction><ode><mu>0.6</mu><mu2>0.6</mu2></ode></friction>
        </surface>
      </collision>{extra}
    </link>
  </model>
</sdf>
""")
    print(f'  model {name}: visual {nv} verts / {nf} tris, '
          f'collision {cnv} verts / {cnf} tris')


# ---------------------------------------------------------------------------
# Ground truth export
# ---------------------------------------------------------------------------

def local_tm_proj(lat0, lon0):
    """Transverse Mercator centred on the world origin (== Gazebo ENU to mm
    level over the few-km extents used here)."""
    return (f'+proj=tmerc +lat_0={lat0} +lon_0={lon0} +k=1 +x_0=0 +y_0=0 '
            '+ellps=WGS84 +units=m +no_defs')


def save_ground_truth(gt_dir, name, Z, dx, x_west, y_north, lat0, lon0,
                      meta=None, masks=None):
    os.makedirs(gt_dir, exist_ok=True)
    base = os.path.join(gt_dir, name)
    ny, nx = Z.shape
    info = {
        'name': name,
        'frame': 'Gazebo world ENU (x east, y north, z up), metres',
        'z_reference': 'still-water level (VRX fluid_level = 0)',
        'shape_rows_cols': [ny, nx],
        'dx_m': dx,
        'x_west_m': x_west,
        'y_north_m': y_north,
        'note': 'Z[row, col] is at x = x_west + col*dx, '
                'y = y_north - row*dx',
        'origin_lat_deg': lat0,
        'origin_lon_deg': lon0,
        'geotiff_crs_proj4': local_tm_proj(lat0, lon0),
    }
    info.update(meta or {})
    arrays = {'z': Z.astype(np.float32)}
    for k, m in (masks or {}).items():
        if m is not None:
            arrays[f'mask_{k}'] = m
    np.savez_compressed(base + '.npz', **arrays)
    with open(base + '.json', 'w') as f:
        json.dump(info, f, indent=2)

    try:
        import rasterio
        from rasterio.transform import from_origin
        with rasterio.open(
                base + '.tif', 'w', driver='GTiff', height=ny, width=nx,
                count=1, dtype='float32', compress='deflate',
                crs=local_tm_proj(lat0, lon0),
                # pixel-is-area: shift half a cell so centres match vertices
                transform=from_origin(x_west - dx / 2, y_north + dx / 2,
                                      dx, dx)) as dst:
            dst.write(Z.astype(np.float32), 1)
    except ImportError:
        pass


def save_preview(path, Z, dx, x_west, y_north, title='', diff=False,
                 vlim=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ny, nx = Z.shape
    ext = [x_west, x_west + dx * (nx - 1), y_north - dx * (ny - 1), y_north]
    fig, ax = plt.subplots(figsize=(10, 10 * ny / nx + 1))
    if diff:
        v = vlim or float(np.nanpercentile(np.abs(Z), 99.5) or 0.1)
        im = ax.imshow(Z, cmap='RdBu', vmin=-v, vmax=v, extent=ext)
        plt.colorbar(im, ax=ax, label='dz [m] (red: erosion, blue: deposition)',
                     shrink=0.7)
    else:
        lo, hi = vlim or (float(np.nanmin(Z)), float(np.nanmax(Z)))
        im = ax.imshow(Z, cmap='terrain', vmin=lo, vmax=hi, extent=ext)
        ax.contour(np.linspace(ext[0], ext[1], nx),
                   np.linspace(ext[3], ext[2], ny), Z, levels=[0],
                   colors='k', linewidths=0.8)
        plt.colorbar(im, ax=ax, label='z [m]', shrink=0.7)
    ax.set_xlabel('x east [m]')
    ax.set_ylabel('y north [m]')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------------------
# World SDF
# ---------------------------------------------------------------------------

GUI = """
    <gui fullscreen="0">
      <plugin filename="MinimalScene" name="3D View">
        <gz-gui>
          <title>3D View</title>
          <property type="bool" key="showTitleBar">false</property>
          <property type="string" key="state">docked</property>
        </gz-gui>
        <engine>ogre2</engine>
        <scene>scene</scene>
        <ambient_light>0.4 0.4 0.4</ambient_light>
        <background_color>0.8 0.8 0.8</background_color>
        <camera_pose>{camera_pose}</camera_pose>
        <camera_clip><near>0.25</near><far>10000</far></camera_clip>
      </plugin>
      <plugin filename="GzSceneManager" name="Scene Manager">
        <gz-gui><property key="state" type="string">floating</property>
          <property key="width" type="double">5</property><property key="height" type="double">5</property>
          <property key="showTitleBar" type="bool">false</property></gz-gui>
      </plugin>
      <plugin filename="InteractiveViewControl" name="Interactive view control">
        <gz-gui><property key="state" type="string">floating</property>
          <property key="width" type="double">5</property><property key="height" type="double">5</property>
          <property key="showTitleBar" type="bool">false</property></gz-gui>
      </plugin>
      <plugin filename="CameraTracking" name="Camera Tracking">
        <gz-gui><property key="state" type="string">floating</property>
          <property key="width" type="double">5</property><property key="height" type="double">5</property>
          <property key="showTitleBar" type="bool">false</property></gz-gui>
      </plugin>
      <plugin filename="MarkerManager" name="Marker manager">
        <gz-gui><property key="state" type="string">floating</property>
          <property key="width" type="double">5</property><property key="height" type="double">5</property>
          <property key="showTitleBar" type="bool">false</property></gz-gui>
      </plugin>
      <plugin filename="SelectEntities" name="Select Entities">
        <gz-gui><property key="state" type="string">floating</property>
          <property key="width" type="double">5</property><property key="height" type="double">5</property>
          <property key="showTitleBar" type="bool">false</property></gz-gui>
      </plugin>
      <plugin filename="VisualizationCapabilities" name="Visualization Capabilities">
        <gz-gui><property key="state" type="string">floating</property>
          <property key="width" type="double">5</property><property key="height" type="double">5</property>
          <property key="showTitleBar" type="bool">false</property></gz-gui>
      </plugin>
      <plugin filename="EntityContextMenuPlugin" name="Entity context menu">
        <gz-gui><property key="state" type="string">floating</property>
          <property key="width" type="double">5</property><property key="height" type="double">5</property>
          <property key="showTitleBar" type="bool">false</property></gz-gui>
      </plugin>
      <plugin filename="WorldControl" name="World control">
        <gz-gui>
          <title>World control</title>
          <property type="bool" key="showTitleBar">false</property>
          <property type="bool" key="resizable">false</property>
          <property type="double" key="height">72</property>
          <property type="double" key="width">121</property>
          <property type="double" key="z">1</property>
          <property type="string" key="state">floating</property>
          <anchors target="3D View"><line own="left" target="left"/><line own="bottom" target="bottom"/></anchors>
        </gz-gui>
        <play_pause>true</play_pause>
        <step>true</step>
        <start_paused>true</start_paused>
        <use_event>true</use_event>
      </plugin>
      <plugin filename="WorldStats" name="World stats">
        <gz-gui>
          <title>World stats</title>
          <property type="bool" key="showTitleBar">false</property>
          <property type="bool" key="resizable">false</property>
          <property type="double" key="height">110</property>
          <property type="double" key="width">290</property>
          <property type="double" key="z">1</property>
          <property type="string" key="state">floating</property>
          <anchors target="3D View"><line own="right" target="right"/><line own="bottom" target="bottom"/></anchors>
        </gz-gui>
        <sim_time>true</sim_time>
        <real_time>true</real_time>
        <real_time_factor>true</real_time_factor>
        <iterations>true</iterations>
      </plugin>
      <plugin filename="TransformControl" name="Transform control"><legacy>false</legacy></plugin>
      <plugin filename="Screenshot" name="Screenshot"></plugin>
      <plugin filename="ComponentInspector" name="Component inspector"></plugin>
      <plugin filename="EntityTree" name="Entity tree"></plugin>
      <plugin filename="ViewAngle" name="View angle"><legacy>false</legacy></plugin>
    </gui>
"""

SYSTEMS = """
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"></plugin>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"></plugin>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"></plugin>
    <plugin filename="gz-sim-magnetometer-system" name="gz::sim::systems::Magnetometer"></plugin>
    <plugin filename="gz-sim-forcetorque-system" name="gz::sim::systems::ForceTorque"></plugin>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"></plugin>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"></plugin>
    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"></plugin>
"""


def _param(key, value):
    return f"""        params {{
          key: "{key}"
          value {{ type: DOUBLE double_value: {value:.6f} }}
        }}
"""


def write_world(path, name, lat, lon, elevation, includes, waves, wind_speed=0.0,
                wind_dir_deg=0.0, camera_pose='0 -300 120 0 0.4 1.57',
                comment=''):
    """includes: list of raw SDF snippets (strings)."""
    wp = ''.join(_param(k, waves[k]) for k in
                 ('direction', 'gain', 'period', 'steepness'))
    body = '\n'.join(includes)
    sdf = f"""<?xml version="1.0" ?>
<!--
  {name}: generated by vrx_gz/worlds/coastal/scripts (do not edit by hand; re-run the
  generator instead).
  {comment}
  Sea state: Hs ~= {waves['hs']:.2f} m, Tp = {waves['period']:.1f} s,
  propagation direction = {math.degrees(waves['direction']):.0f} deg (ENU).
-->
<sdf version="1.9">
  <world name="{name}">

    <physics name="4ms" type="dart">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
{GUI.format(camera_pose=camera_pose)}
{SYSTEMS}
    <scene>
      <sky></sky>
      <grid>false</grid>
      <ambient>1.0 1.0 1.0</ambient>
      <background>0.8 0.8 0.8</background>
    </scene>

    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>{lat:.7f}</latitude_deg>
      <longitude_deg>{lon:.7f}</longitude_deg>
      <elevation>{elevation:.2f}</elevation>
      <heading_deg>0.0</heading_deg>
    </spherical_coordinates>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation><range>1000</range><constant>0.9</constant>
        <linear>0.01</linear><quadratic>0.001</quadratic></attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <include>
      <name>Coast Waves</name>
      <pose>0 0 0 0 0 0</pose>
      <uri>coast_waves</uri>
    </include>

{body}

    <plugin filename="libUSVWind.so" name="vrx::USVWind">
      <wind_obj>
        <name>wamv</name>
        <link_name>wamv/base_link</link_name>
        <coeff_vector>.5 .5 .33</coeff_vector>
      </wind_obj>
      <wind_direction>{wind_dir_deg}</wind_direction>
      <wind_mean_velocity>{wind_speed}</wind_mean_velocity>
      <var_wind_gain_constants>0</var_wind_gain_constants>
      <var_wind_time_constants>2</var_wind_time_constants>
      <random_seed>10</random_seed>
      <update_rate>10</update_rate>
      <topic_wind_speed>/vrx/debug/wind/speed</topic_wind_speed>
      <topic_wind_direction>/vrx/debug/wind/direction</topic_wind_direction>
    </plugin>

    <plugin filename="libPublisherPlugin.so" name="vrx::PublisherPlugin">
      <message type="gz.msgs.Param" topic="/vrx/wavefield/parameters" every="2.0">
{wp}      </message>
    </plugin>

  </world>
</sdf>
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(sdf)


def include(uri, name, pose='0 0 0 0 0 0'):
    return f"""    <include>
      <name>{name}</name>
      <pose>{pose}</pose>
      <uri>{uri}</uri>
    </include>"""
