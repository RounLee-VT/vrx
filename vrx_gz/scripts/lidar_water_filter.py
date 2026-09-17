#!/usr/bin/env python3
"""Drop LiDAR returns that lie below the water surface or on the own vessel.

VRX LiDARs use visibility_mask 7, which hides the wave visual from GPU ray
sensors. As a result, simulated beams pass through the water and return the
seabed or the underwater parts of structures, which a real near-IR LiDAR does
not (water absorbs the beam and the surface gives at most sparse specular
returns). This node transforms each cloud into the world frame and removes
points with z < water_level + margin, then republishes it in the original
sensor frame with all fields intact.

Parameters:
  input_topic, output_topic
  world_frame   (default 'world')
  water_level   (default 0.0, world z of still water)
  margin        (default 0.0; > 0 also removes points in the wave band)
  self_frame    (default 'wamv/wamv/base_link')
  self_box_min / self_box_max  (vessel crop box in self_frame, metres;
                default WAM-V hull + deck equipment)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
import tf2_ros


def quat_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class LidarWaterFilter(Node):
    def __init__(self):
        super().__init__('lidar_water_filter')
        self.declare_parameter('input_topic', '/wamv/sensors/lidars/lidar_wamv_sensor/points')
        self.declare_parameter('output_topic',
                               '/wamv/sensors/lidars/lidar_wamv_sensor/points_filtered')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('water_level', 0.0)
        self.declare_parameter('margin', 0.0)
        self.declare_parameter('self_frame', 'wamv/wamv/base_link')
        self.declare_parameter('self_box_min', [-2.7, -1.4, -0.6])
        self.declare_parameter('self_box_max', [2.7, 1.4, 2.25])
        self.world = self.get_parameter('world_frame').value
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # Reliable output: compatible with both reliable (RViz default) and
        # best-effort subscribers.
        self.pub = self.create_publisher(
            PointCloud2, self.get_parameter('output_topic').value, 5)
        self.create_subscription(
            PointCloud2, self.get_parameter('input_topic').value, self.cb,
            qos_profile_sensor_data)
        self.warned = False

    def cb(self, msg):
        try:
            # Transform at the scan time: the hull pitches/rolls in waves, and a
            # 1 deg attitude error is ~1.7 m in height at 100 m range.
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.world, msg.header.frame_id, Time.from_msg(msg.header.stamp),
                    timeout=Duration(seconds=0.05))
            except tf2_ros.ExtrapolationException:
                tf = self.tf_buffer.lookup_transform(self.world, msg.header.frame_id, Time())
        except tf2_ros.TransformException as e:
            if not self.warned:
                self.get_logger().warn(f'waiting for TF {self.world} <- '
                                       f'{msg.header.frame_id}: {e}')
                self.warned = True
            return
        pts = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=False)
        xyz = np.stack([pts['x'], pts['y'], pts['z']], axis=-1).astype(np.float64)
        finite = np.isfinite(xyz).all(axis=1)
        t = tf.transform.translation
        xyz0 = np.where(finite[:, None], xyz, 0.0)
        zw = np.where(finite, xyz0 @ quat_to_matrix(tf.transform.rotation)[2] + t.z, -np.inf)
        level = self.get_parameter('water_level').value + self.get_parameter('margin').value
        keep = finite & (zw >= level)
        try:
            st = self.tf_buffer.lookup_transform(self.get_parameter('self_frame').value,
                                                 msg.header.frame_id, Time())
            p = xyz0 @ quat_to_matrix(st.transform.rotation).T + np.array(
                [st.transform.translation.x, st.transform.translation.y,
                 st.transform.translation.z])
            lo = np.array(self.get_parameter('self_box_min').value)
            hi = np.array(self.get_parameter('self_box_max').value)
            keep &= ~np.all((p >= lo) & (p <= hi), axis=1)
        except tf2_ros.TransformException:
            pass
        # Select whole point records from the raw buffer: keeps every field
        # (intensity, ring, padding) exactly as published.
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
        out = PointCloud2()
        out.header = msg.header
        out.height = 1
        out.width = int(keep.sum())
        out.fields = msg.fields
        out.is_bigendian = msg.is_bigendian
        out.point_step = msg.point_step
        out.row_step = msg.point_step * out.width
        out.is_dense = True
        out.data = raw[keep].tobytes()
        self.pub.publish(out)


def main():
    rclpy.init()
    node = LidarWaterFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
