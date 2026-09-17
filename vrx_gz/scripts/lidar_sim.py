#!/usr/bin/env python3
"""LiDAR post-processing for the coastal survey worlds.

Gazebo GPU LiDARs return geometry only: their `intensity` channel is a single
per-visual laser_retro value (0 here), and VRX LiDARs ignore the wave visual
(visibility_mask 7), so beams pass through the water and return the seabed.
This node turns the raw organised cloud into a more realistic sensor output:

1. Water: removes returns below the water level (world z < water_level +
   margin), using the TF at scan time. A real near-IR LiDAR gets no returns
   through water.
2. Own vessel: removes hits inside a crop box around the hull.
3. Intensity (Velodyne-style calibrated reflectivity, 0-100 for diffuse
   targets):
       intensity = 100 * rho(material) * cos(theta_i) * (1 + noise)
   rho comes from the world's material map (vrx_gz/coastal_materials.py,
   config/coastal_materials.yaml). theta_i is the incidence angle, from the
   local surface normal estimated on the organised scan grid, falling back
   to the ground-truth DEM normal for grazing hits on terrain.
4. Detection limit: drops weak returns, where
       rho * cos(theta_i) * (ref_range / r)^2 < min_reflectivity
   (by default a 10 % target is detected out to 100 m), so dark or grazing
   surfaces drop out at long range.

Output fields: x y z intensity signal_db ring material
  signal_db  10 log10(rho cos(theta_i) / r^2), range-dependent return strength
  material   material id (ground-truth label, see coastal_materials.yaml)
"""

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
import tf2_ros

from vrx_gz.coastal_materials import MaterialMap

OUT_DTYPE = np.dtype({
    'names': ['x', 'y', 'z', 'intensity', 'signal_db', 'ring', 'material'],
    'formats': ['<f4', '<f4', '<f4', '<f4', '<f4', '<u2', 'u1'],
    'offsets': [0, 4, 8, 12, 16, 20, 22], 'itemsize': 24})
OUT_FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name='signal_db', offset=16, datatype=PointField.FLOAT32, count=1),
    PointField(name='ring', offset=20, datatype=PointField.UINT16, count=1),
    PointField(name='material', offset=22, datatype=PointField.UINT8, count=1),
]


def quat_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def grid_normals(P, valid, rel_tol=0.1, abs_tol=0.5, wrap_cols=True):
    """Surface normals (sensor frame) on an organised (rows, cols, 3) grid.

    Uses central differences, falling back to one-sided ones; a neighbour is
    rejected when its range differs by more than max(abs_tol, rel_tol * r)
    (depth discontinuity). Returns (normals, ok)."""
    R = np.linalg.norm(P, axis=-1)

    def tangent(axis, wrap):
        def shift(a, k):
            if wrap:
                return np.roll(a, -k, axis=axis)
            out = np.roll(a, -k, axis=axis)
            idx = [slice(None)] * a.ndim
            idx[axis] = slice(-k, None) if k > 0 else slice(None, -k)
            out[tuple(idx)] = np.nan if a.dtype.kind == 'f' else False
            return out
        Pf, Pb = shift(P, 1), shift(P, -1)
        Rf, Rb = shift(R, 1), shift(R, -1)
        vf, vb = shift(valid, 1), shift(valid, -1)
        tol = np.maximum(abs_tol, rel_tol * R)
        okf = valid & vf & (np.abs(Rf - R) <= tol)
        okb = valid & vb & (np.abs(Rb - R) <= tol)
        t = np.where((okf & okb)[..., None], Pf - Pb,
                     np.where(okf[..., None], Pf - P, np.where(okb[..., None], P - Pb, 0.0)))
        return np.nan_to_num(t), okf | okb

    th, okh = tangent(1, wrap_cols)
    tv, okv = tangent(0, False)
    n = np.cross(th, tv)
    nn = np.linalg.norm(n, axis=-1)
    ok = okh & okv & (nn > 1e-9)
    return n / np.maximum(nn, 1e-12)[..., None], ok


class LidarSim(Node):
    def __init__(self):
        super().__init__('lidar_sim')
        p = self.declare_parameter
        p('input_topic', '/wamv/sensors/lidars/lidar_wamv_sensor/points')
        p('output_topic', '/wamv/sensors/lidars/lidar_wamv_sensor/points_filtered')
        p('world', '')
        p('world_frame', 'world')
        p('water_level', 0.0)
        p('margin', 0.0)
        p('self_frame', 'wamv/wamv/base_link')
        p('self_box_min', [-2.7, -1.4, -0.6])
        p('self_box_max', [2.7, 1.4, 2.25])
        p('reflectivity_noise', 0.05)          # relative (1 sigma)
        p('detection_ref_range', 100.0)
        p('detection_min_reflectivity', 0.10)
        p('default_incidence_cos', 0.7)
        p('seed', 0)
        self.g = lambda k: self.get_parameter(k).value  # noqa: E731
        self.rng = np.random.default_rng(self.g('seed') or None)
        self.materials = MaterialMap(self.g('world'))
        if not self.materials.available:
            self.get_logger().warn(f"no material map for world '{self.g('world')}': "
                                   "using 'unknown' material everywhere")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # Reliable: compatible with both reliable (RViz) and best-effort subscribers.
        self.pub = self.create_publisher(PointCloud2, self.g('output_topic'), 5)
        self.create_subscription(PointCloud2, self.g('input_topic'), self.cb,
                                 qos_profile_sensor_data)
        self.warned = False

    def lookup(self, target, source, stamp):
        # Transform at the scan time: the hull pitches/rolls in waves, and a
        # 1 deg attitude error is ~1.7 m in height at 100 m range.
        try:
            tf = self.tf_buffer.lookup_transform(target, source, Time.from_msg(stamp),
                                                 timeout=Duration(seconds=0.05))
        except tf2_ros.ExtrapolationException:
            tf = self.tf_buffer.lookup_transform(target, source, Time())
        t = tf.transform.translation
        return quat_to_matrix(tf.transform.rotation), np.array([t.x, t.y, t.z])

    def cb(self, msg):
        g = self.g
        try:
            Rw, tw = self.lookup(g('world_frame'), msg.header.frame_id, msg.header.stamp)
        except tf2_ros.TransformException as e:
            if not self.warned:
                self.get_logger().warn(f"waiting for TF {g('world_frame')} <- "
                                       f'{msg.header.frame_id}: {e}')
                self.warned = True
            return
        names = [f.name for f in msg.fields]
        raw = pc2.read_points(msg, skip_nans=False)
        rows = max(msg.height, 1)
        P = np.stack([raw['x'], raw['y'], raw['z']], -1).astype(np.float64).reshape(rows, -1, 3)
        ring = (np.asarray(raw['ring']).reshape(rows, -1) if 'ring' in names
                else np.repeat(np.arange(rows)[:, None], P.shape[1], 1))
        R = np.linalg.norm(P, axis=-1)
        valid = np.isfinite(P).all(-1) & (R > 0)
        P0 = np.where(valid[..., None], P, 0.0)

        Pw = P0 @ Rw.T + tw
        keep = valid & (Pw[..., 2] >= g('water_level') + g('margin'))
        try:
            Rs, ts = self.lookup(g('self_frame'), msg.header.frame_id, msg.header.stamp)
            Ps = P0 @ Rs.T + ts
            lo, hi = np.array(g('self_box_min')), np.array(g('self_box_max'))
            keep &= ~np.all((Ps >= lo) & (Ps <= hi), axis=-1)
        except tf2_ros.TransformException:
            pass

        # incidence angle: scan-grid normal, else ground-truth DEM normal
        n_grid, ok_grid = grid_normals(P0, valid)
        idx = np.flatnonzero(keep)
        u = P0.reshape(-1, 3)[idx] / R.reshape(-1)[idx, None]
        ids, n_dem_w = self.materials.classify(Pw.reshape(-1, 3)[idx])
        cos_i = np.full(len(idx), g('default_incidence_cos'))
        has_dem = np.isfinite(n_dem_w).all(axis=1)
        cos_i[has_dem] = np.abs(np.einsum('ij,ij->i', n_dem_w[has_dem] @ Rw, u[has_dem]))
        og = ok_grid.reshape(-1)[idx]
        cos_i[og] = np.abs(np.einsum('ij,ij->i', n_grid.reshape(-1, 3)[idx][og], u[og]))
        cos_i = np.clip(cos_i, 0.05, 1.0)

        rho = self.materials.table.reflectivity[ids]
        r = R.reshape(-1)[idx]
        detect = rho * cos_i * (g('detection_ref_range') / r) ** 2 >= g('detection_min_reflectivity')
        noise = 1.0 + self.rng.normal(0.0, g('reflectivity_noise'), len(idx))
        intensity = np.clip(100.0 * rho * cos_i * noise, 0.0, 255.0)
        signal_db = 10.0 * np.log10(rho * cos_i / r ** 2)

        sel = detect
        out = np.zeros(int(sel.sum()), dtype=OUT_DTYPE)
        pts = P0.reshape(-1, 3)[idx][sel]
        out['x'], out['y'], out['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
        out['intensity'] = intensity[sel]
        out['signal_db'] = signal_db[sel]
        out['ring'] = ring.reshape(-1)[idx][sel]
        out['material'] = ids[sel]

        cloud = PointCloud2()
        cloud.header = msg.header
        cloud.height, cloud.width = 1, len(out)
        cloud.fields = OUT_FIELDS
        cloud.is_bigendian = False
        cloud.point_step = OUT_DTYPE.itemsize
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = True
        cloud.data = out.tobytes()
        self.pub.publish(cloud)


def main():
    rclpy.init()
    node = LidarSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
