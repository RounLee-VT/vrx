#!/usr/bin/env python3
"""Omniscan 3D 450 SS output emulation on top of the Gazebo ray-fan proxy.

Input: the raw GPU-ray point cloud of `omniscan3d_sensor` (ideal geometry, no
noise), organised as rows (along-track angles across the 0.8 deg beam) x
columns (cross-track beam angles across the 90 deg fan). Sensor frame:
x = boresight, y = starboard, z = along-track.

Per ping this node:
  1. Along-track beam width: merges the rows of each column into one echo,
     weighted by the along-track beam pattern (-3 dB at +/- beamwidth/2),
     like a real 0.8 deg beam averaging its footprint ('first' mode keeps the
     leading edge instead).
  2. Time of flight: tof = 2 r / c (two-way; see note below).
  3. Power (relative dB), sonar equation:
       pwr = SL - 2 TL(r) + BS(theta_i) + B_tx(angle) + speckle
       TL = 20 log10 r + alpha r      (spherical spreading + absorption)
       BS = mu(material) + 10 log10 cos^2 theta_i
            (Lambert's law; mu per material from the world's material map,
             theta_i from the local surface normal of the raw grid)
       B_tx = -10 dB * (angle / 45 deg)^2 (90 deg TX beam, -10 dB edge)
       speckle: Rayleigh amplitude -> exponentially distributed power
  4. Range noise and quantisation, then drops echoes below the noise floor.

Outputs:
  ~/points (PointCloud2, sensor frame): x y z angle tof pwr pt_type material
      (material: ground-truth label, see config/coastal_materials.yaml)
  ~/os3d_point_set (std_msgs/UInt8MultiArray): one Cerulean Ping Protocol
      packet per ping, packet id 3104, payload laid out as documented in
      docs.ceruleansonar.com/c/omniscan3d/technical-details/api/os3d_point_set

Assumptions (not specified by the vendor docs): `tof` is two-way; `pwr`
absolute scale and the three thresholds are simulation conventions
(floor + 6/12/20 dB), not calibrated to the hardware. Byte order is
little-endian, as in the Ping protocol implementations.
"""

import struct
import warnings

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import UInt8MultiArray
import tf2_ros

from vrx_gz.coastal_materials import MaterialMap

OS3D_POINT_SET_ID = 3104
HEADER_FMT = '<IfHHIQIBBBBfff9I'          # 80 bytes, see vendor API page
ATOF_FMT = '<fffB3x'                      # 16 bytes: angle, tof, pwr, pt_type
HEADER_SIZE = struct.calcsize(HEADER_FMT)
ATOF_SIZE = struct.calcsize(ATOF_FMT)


def encode_packet(packet_id, payload, src=0, dst=0):
    """Cerulean Ping Protocol framing: 'BR', len, id, src, dst, payload, sum16."""
    head = struct.pack('<2sHHBB', b'BR', len(payload), packet_id, src, dst)
    body = head + payload
    return body + struct.pack('<H', sum(body) & 0xFFFF)


def encode_os3d_point_set(ping_number, sos, utc_msec, pwr_up_msec, thresholds,
                          angle, tof, pwr, pt_type, device=0):
    n = len(angle)
    header = struct.pack(HEADER_FMT, ping_number, sos, n, 0, 0, utc_msec,
                         pwr_up_msec, 1, device, 0, 0, *thresholds, *([0] * 9))
    pts = np.zeros(n, dtype=np.dtype([('angle', '<f4'), ('tof', '<f4'),
                                      ('pwr', '<f4'), ('pt_type', 'u1'),
                                      ('res', 'u1', 3)]))
    pts['angle'], pts['tof'], pts['pwr'], pts['pt_type'] = angle, tof, pwr, pt_type
    return encode_packet(OS3D_POINT_SET_ID, header + pts.tobytes())


def decode_os3d_point_set(packet):
    """Inverse of encode_os3d_point_set (for tests and consumers)."""
    packet = bytes(packet)
    magic, n_payload, pid, src, dst = struct.unpack_from('<2sHHBB', packet)
    assert magic == b'BR' and pid == OS3D_POINT_SET_ID
    assert struct.unpack_from('<H', packet, 8 + n_payload)[0] == \
        sum(packet[:8 + n_payload]) & 0xFFFF, 'checksum'
    h = struct.unpack_from(HEADER_FMT, packet, 8)
    n = h[2]
    pts = np.frombuffer(packet, dtype=np.dtype([('angle', '<f4'), ('tof', '<f4'),
                                                ('pwr', '<f4'), ('pt_type', 'u1'),
                                                ('res', 'u1', 3)]),
                        count=n, offset=8 + HEADER_SIZE)
    return {'ping_number': h[0], 'sos_mps': h[1], 'num_points': n,
            'utc_msec': h[5], 'pwr_up_msec': h[6], 'version': h[7],
            'device': h[8], 'pwr_threshold_high': h[11],
            'pwr_threshold_med': h[12], 'pwr_threshold_low': h[13],
            'angle': pts['angle'], 'tof': pts['tof'], 'pwr': pts['pwr'],
            'pt_type': pts['pt_type']}


class Omniscan3DSim(Node):
    def __init__(self):
        super().__init__('omniscan3d_sim')
        p = self.declare_parameter
        p('input_topic', '/wamv/sensors/lidars/omniscan3d_sensor/points')
        p('output_topic', '/wamv/sensors/sonars/omniscan3d/points')
        p('packet_topic', '/wamv/sensors/sonars/omniscan3d/os3d_point_set')
        p('sound_speed_mps', 1500.0)
        p('absorption_db_per_m', 0.12)       # ~seawater @450 kHz; fresh ~0.05
        p('source_level_db', 0.0)
        p('world', '')
        p('world_frame', 'world')
        p('backscatter_mu_db', -27.0)        # used where no material map exists
        p('tx_edge_loss_db', 10.0)           # at +/- tx_half_angle_deg
        p('tx_half_angle_deg', 45.0)
        p('along_track_beamwidth_deg', 0.8)
        p('along_track_mode', 'weighted')    # weighted | first
        p('noise_floor_db', -110.0)
        p('speckle', True)
        p('range_noise_std_m', 0.02)
        p('range_resolution_m', 0.02)
        p('tof_two_way', True)
        p('device_number', 0)
        p('seed', 0)
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        self.g = g
        self.rng = np.random.default_rng(g('seed') or None)
        self.ping = 0
        self.t0_ms = None
        self.materials = MaterialMap(g('world'))
        if not self.materials.available:
            self.get_logger().warn(f"no material map for world '{g('world')}': "
                                   f"using backscatter_mu_db={g('backscatter_mu_db')}")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(PointCloud2, g('output_topic'), 5)
        self.pub_pkt = self.create_publisher(UInt8MultiArray, g('packet_topic'), 5)
        self.create_subscription(PointCloud2, g('input_topic'), self.cb,
                                 qos_profile_sensor_data)
        self.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='angle', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='tof', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='pwr', offset=20, datatype=PointField.FLOAT32, count=1),
            PointField(name='pt_type', offset=24, datatype=PointField.UINT8, count=1),
            PointField(name='material', offset=25, datatype=PointField.UINT8, count=1),
        ]
        self.dtype = np.dtype({'names': ['x', 'y', 'z', 'angle', 'tof', 'pwr', 'pt_type',
                                         'material'],
                               'formats': ['<f4'] * 6 + ['u1', 'u1'],
                               'offsets': [0, 4, 8, 12, 16, 20, 24, 25], 'itemsize': 28})

    # ------------------------------------------------------------------
    def cb(self, msg):
        g = self.g
        raw = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=False)
        P = np.stack([raw['x'], raw['y'], raw['z']], -1).astype(np.float64)
        rows = max(msg.height, 1)
        P = P.reshape(rows, -1, 3)                       # (rows, cols, 3)
        R = np.linalg.norm(P, axis=-1)
        valid = np.isfinite(R) & (R > 0)
        Pz = np.where(valid[..., None], P, 0.0)

        # beam angles from the geometry of each ray
        with np.errstate(invalid='ignore', divide='ignore'):
            psi = np.degrees(np.arctan2(Pz[..., 2], np.hypot(Pz[..., 0], Pz[..., 1])))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN columns
            col_angle = np.nanmedian(
                np.where(valid, np.arctan2(Pz[..., 1], Pz[..., 0]), np.nan), axis=0)

        # 1. along-track beam: merge rows per column
        half = g('along_track_beamwidth_deg') / 2.0
        w = np.where(valid, 10 ** (-3.0 * (psi / half) ** 2 / 10.0), 0.0) if half > 0 \
            else valid.astype(float)
        if g('along_track_mode') == 'first':
            r = np.where(valid, R, np.inf).min(axis=0)
            ok = np.isfinite(r)
        else:
            wsum = w.sum(axis=0)
            ok = wsum > 0
            r = np.where(ok, (w * np.where(valid, R, 0)).sum(axis=0) / np.maximum(wsum, 1e-12), np.nan)
        ok &= np.isfinite(col_angle)

        # incidence angle from the local surface normal of the raw grid
        cos_i = self.incidence_cos(Pz, valid)

        # material of each echo (world-frame lookup at the beam centre)
        mat = np.full(r.shape, self.materials.unknown_id, np.uint8)
        mu = np.full(r.shape, g('backscatter_mu_db'))
        if self.materials.available:
            try:
                # Non-blocking: at 20 Hz a wait would drop pings; the latest TF
                # is at most a few ms off, negligible for a material lookup.
                try:
                    tf = self.tf_buffer.lookup_transform(
                        g('world_frame'), msg.header.frame_id,
                        Time.from_msg(msg.header.stamp))
                except tf2_ros.ExtrapolationException:
                    tf = self.tf_buffer.lookup_transform(g('world_frame'),
                                                         msg.header.frame_id, Time())
                q, t = tf.transform.rotation, tf.transform.translation
                x, y, z, w = q.x, q.y, q.z, q.w
                Rw = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                               [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                               [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
                rs = np.nan_to_num(r)
                ps = np.stack([rs * np.cos(np.nan_to_num(col_angle)),
                               rs * np.sin(np.nan_to_num(col_angle)), np.zeros_like(rs)], -1)
                ids, _ = self.materials.classify(ps @ Rw.T + [t.x, t.y, t.z])
                mat = ids
                mu = self.materials.table.mu_db[ids].astype(np.float64)
            except tf2_ros.TransformException:
                pass

        # 2./3. sonar equation
        with np.errstate(invalid='ignore', divide='ignore'):
            tl = 20.0 * np.log10(np.maximum(r, 0.1)) + g('absorption_db_per_m') * r
            bs = mu + 10.0 * np.log10(np.maximum(cos_i ** 2, 1e-4))
            ang_deg = np.degrees(col_angle)
            btx = -g('tx_edge_loss_db') * (ang_deg / g('tx_half_angle_deg')) ** 2
            pwr = g('source_level_db') - 2.0 * tl + bs + btx
            if g('speckle'):
                pwr = pwr + 10.0 * np.log10(np.maximum(self.rng.exponential(1.0, pwr.shape), 1e-6))
        floor = g('noise_floor_db')
        ok &= np.isfinite(pwr) & (pwr >= floor)

        # 4. range noise + quantisation
        rn = r + self.rng.normal(0.0, g('range_noise_std_m'), r.shape)
        res = g('range_resolution_m')
        if res > 0:
            rn = np.round(rn / res) * res
        rn = np.maximum(rn, 0.0)

        a, rr, pw, mt = col_angle[ok], rn[ok], pwr[ok], mat[ok]
        sos = g('sound_speed_mps')
        tof = (2.0 if g('tof_two_way') else 1.0) * rr / sos

        pts = np.zeros(len(a), dtype=self.dtype)
        pts['x'], pts['y'], pts['z'] = rr * np.cos(a), rr * np.sin(a), 0.0
        pts['angle'], pts['tof'], pts['pwr'], pts['pt_type'] = a, tof, pw, 0
        pts['material'] = mt
        out = PointCloud2()
        out.header = msg.header
        out.height, out.width = 1, len(pts)
        out.fields = self.fields
        out.is_bigendian = False
        out.point_step = self.dtype.itemsize
        out.row_step = out.point_step * out.width
        out.is_dense = True
        out.data = pts.tobytes()
        self.pub.publish(out)

        stamp_ms = msg.header.stamp.sec * 1000 + msg.header.stamp.nanosec // 1_000_000
        if self.t0_ms is None:
            self.t0_ms = stamp_ms
        thresholds = (floor + 20.0, floor + 12.0, floor + 6.0)     # high, med, low
        pkt = encode_os3d_point_set(self.ping, sos, stamp_ms,
                                    (stamp_ms - self.t0_ms) & 0xFFFFFFFF, thresholds,
                                    a, tof, pw, np.zeros(len(a), np.uint8),
                                    device=g('device_number'))
        self.pub_pkt.publish(UInt8MultiArray(data=list(pkt)))
        self.ping = (self.ping + 1) & 0xFFFFFFFF

    @staticmethod
    def incidence_cos(P, valid):
        """|cos| of the angle between each column's ray and the surface normal,
        from central differences on the (rows, cols) grid, averaged over rows."""
        rows, cols, _ = P.shape
        cos_rows = np.full((rows, cols), np.nan)
        z_axis = np.array([0.0, 0.0, 1.0])
        for k in range(rows):
            tc = np.full((cols, 3), np.nan)
            tc[1:-1] = P[k, 2:] - P[k, :-2]
            vc = np.zeros(cols, bool)
            vc[1:-1] = valid[k, 2:] & valid[k, :-2]
            if rows >= 2:
                k0, k1 = max(k - 1, 0), min(k + 1, rows - 1)   # one-sided at edges
                ta = P[k1] - P[k0]
                va = valid[k1] & valid[k0]
            else:
                ta = np.broadcast_to(z_axis, (cols, 3))
                va = np.ones(cols, bool)
            n = np.cross(np.where(vc[:, None], tc, 0), np.where(va[:, None], ta, z_axis))
            nn = np.linalg.norm(n, axis=1)
            u = P[k] / np.maximum(np.linalg.norm(P[k], axis=1, keepdims=True), 1e-9)
            c = np.abs(np.einsum('ij,ij->i', n, u)) / np.maximum(nn, 1e-12)
            c = np.where(vc & valid[k] & (nn > 1e-9), c, np.nan)
            cos_rows[k] = c
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN edge columns
            cos_i = np.nanmean(cos_rows, axis=0) if rows > 1 else cos_rows[0]
        return np.where(np.isfinite(cos_i), np.clip(cos_i, 0.0, 1.0), 1.0)


def main():
    rclpy.init()
    node = Omniscan3DSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
