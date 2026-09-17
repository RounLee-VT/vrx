# Copyright 2026 Virginia Tech.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Surface material lookup for the coastal survey worlds.

Gazebo GPU ray sensors only report geometry (their intensity channel is a
single per-visual laser_retro value, 0 by default), so material-dependent
LiDAR intensity and sonar backscatter are synthesised from per-world material
maps generated alongside the terrain
(worlds/coastal/ground_truth/<world>_materials.npz, see
worlds/coastal/scripts/gen_material_maps.py) and the property table
config/coastal_materials.yaml.

Map types:
  dem    one 2.5D surface: cls, z rasters. A point within `tol` of the
         surface takes the raster class; points clearly above it are
         'object' (targets, pilings, other vessels).
  cliff  ground and plateau rasters plus the 3D cliff face / sea stacks:
         points on neither raster surface are rock, sub-classed by height
         (algae in the intertidal band, wet rock in the splash zone).
"""

import os

import numpy as np
import yaml

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:  # pragma: no cover
    get_package_share_directory = None


class MaterialTable:
    def __init__(self, path):
        with open(path) as f:
            mats = yaml.safe_load(f)['materials']
        n = max(m['id'] for m in mats) + 1
        self.names = [''] * n
        self.reflectivity = np.zeros(n, np.float32)
        self.mu_db = np.zeros(n, np.float32)
        for m in mats:
            self.names[m['id']] = m['name']
            self.reflectivity[m['id']] = m['lidar_reflectivity']
            self.mu_db[m['id']] = m['sonar_mu_db']
        self.ids = {name: i for i, name in enumerate(self.names)}

    def id(self, name):
        return self.ids[name]


class _Raster:
    def __init__(self, z, cls, valid, dx, x_west, y_north):
        self.z = np.asarray(z, np.float64)
        self.cls = np.asarray(cls, np.uint8)
        self.valid = np.ones(self.z.shape, bool) if valid is None else np.asarray(valid, bool)
        self.valid &= np.isfinite(self.z)
        self.dx, self.x_west, self.y_north = float(dx), float(x_west), float(y_north)
        zf = np.where(self.valid, self.z, np.nanmean(self.z[self.valid]) if self.valid.any() else 0)
        gy, gx = np.gradient(zf, self.dx)
        self.gx, self.gy = gx, -gy          # d z / d x_east, d z / d y_north
        self.slope = np.hypot(gx, gy)

    def sample(self, x, y):
        """Nearest-cell lookup; returns (inside, z_bilinear, cls, valid, normal, slope)."""
        ny, nx = self.z.shape
        c = (x - self.x_west) / self.dx
        r = (self.y_north - y) / self.dx
        inside = (c >= 0) & (c <= nx - 1) & (r >= 0) & (r <= ny - 1)
        ci = np.clip(np.round(c).astype(int), 0, nx - 1)
        ri = np.clip(np.round(r).astype(int), 0, ny - 1)
        c0 = np.clip(np.floor(c).astype(int), 0, nx - 2)
        r0 = np.clip(np.floor(r).astype(int), 0, ny - 2)
        fc, fr = np.clip(c - c0, 0, 1), np.clip(r - r0, 0, 1)
        z = (self.z[r0, c0] * (1 - fc) * (1 - fr) + self.z[r0, c0 + 1] * fc * (1 - fr)
             + self.z[r0 + 1, c0] * (1 - fc) * fr + self.z[r0 + 1, c0 + 1] * fc * fr)
        valid = inside & self.valid[ri, ci] & np.isfinite(z)
        n = np.stack([-self.gx[ri, ci], -self.gy[ri, ci], np.ones(len(x))], -1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        return inside, z, self.cls[ri, ci], valid, n, self.slope[ri, ci]


class MaterialMap:
    def __init__(self, world, table=None, map_path=None, share_dir=None):
        if share_dir is None and get_package_share_directory is not None:
            share_dir = get_package_share_directory('vrx_gz')
        self.table = table or MaterialTable(
            os.path.join(share_dir, 'config', 'coastal_materials.yaml'))
        map_path = map_path or os.path.join(
            share_dir, 'worlds', 'coastal', 'ground_truth', f'{world}_materials.npz')
        self.kind = None
        self.object_id = self.table.id('object')
        self.unknown_id = self.table.id('unknown')
        if not os.path.exists(map_path):
            return
        d = np.load(map_path)
        self.kind = str(d['kind'])
        self.outside_id = int(d['outside_class'])
        g = (d['dx'], d['x_west'], d['y_north'])
        if self.kind == 'dem':
            self.surface = _Raster(d['z'], d['cls'], None, *g)
        else:
            self.ground = _Raster(d['ground_z'], d['ground_cls'], d['ground_valid'], *g)
            self.plateau = _Raster(d['plateau_z'], d['plateau_cls'], d['plateau_valid'], *g)
            self.rock = self.table.id('rock')
            self.rock_wet = self.table.id('rock_wet')
            self.algae = self.table.id('algae')

    @property
    def available(self):
        return self.kind is not None

    def classify(self, P, base_tol=0.6):
        """P: (N, 3) world points. Returns (material ids uint8, surface normals
        (N, 3) with NaN rows where no surface normal is known)."""
        N = len(P)
        ids = np.full(N, self.unknown_id, np.uint8)
        normals = np.full((N, 3), np.nan)
        if not self.available or N == 0:
            return ids, normals
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        if self.kind == 'dem':
            inside, zs, cls, valid, n, slope = self.surface.sample(x, y)
            tol = base_tol + 1.5 * self.surface.dx * slope
            dz = z - zs
            on = valid & (np.abs(dz) <= tol)
            ids[on] = cls[on]
            normals[on] = n[on]
            above = valid & (dz > tol)
            ids[above] = self.object_id
            below = valid & (dz < -tol)                 # e.g. under an overhang
            ids[below] = cls[below]
            ids[~inside] = self.outside_id
            return ids, normals
        # cliff world
        gi, gz, gcls, gv, gn, gs = self.ground.sample(x, y)
        pi, pz, pcls, pv, pn, ps = self.plateau.sample(x, y)
        tol_g = base_tol + 1.5 * gs
        on_g = gv & (z <= gz + tol_g)          # on (or noisily below) the ground
        on_p = pv & ~on_g & (np.abs(z - pz) <= base_tol + 1.5 * ps)
        ids[on_g], normals[on_g] = gcls[on_g], gn[on_g]
        ids[on_p], normals[on_p] = pcls[on_p], pn[on_p]
        above_p = pv & ~on_g & ~on_p & (z > pz + base_tol + 1.5 * ps)
        ids[above_p] = self.object_id
        face = ~(on_g | on_p | above_p) & (gi | pi)
        ids[face & (z >= 2.6)] = self.rock
        ids[face & (z < 2.6)] = self.rock_wet
        ids[face & (z < 0.9) & (z > -0.3)] = self.algae
        ids[~(gi | pi)] = self.outside_id
        return ids, normals
