"""Launch a coastal survey world (vrx_gz/worlds) with the survey WAM-V.

  ros2 launch vrx_gz coastal_survey.launch.py world:=nbs_surf_epoch0
  ros2 launch vrx_gz coastal_survey.launch.py world:=cliff_coast_epoch0 rviz:=false
  ros2 launch vrx_gz coastal_survey.launch.py world:=ocean_view_norfolk headless:=true
  ros2 launch vrx_gz coastal_survey.launch.py world:=claytor_lake_calm x:=0 y:=0 yaw:=0

Besides the standard VRX bridges this starts:
  * static TF wamv/base_link -> wamv/wamv/base_link, joining the world pose
    tree (pose_tf_broadcaster) with the robot_state_publisher / sensor tree,
    so everything resolves in the 'world' frame
  * /wamv/ground_truth/odometry (nav_msgs/Odometry, world frame, 50 Hz)
  * lidar_water_filter: /wamv/sensors/lidars/lidar_wamv_sensor/points_filtered
    (LiDAR without returns below the water surface)
  * omniscan3d_sim: /wamv/sensors/sonars/omniscan3d/points (x y z angle tof pwr
    pt_type) and .../os3d_point_set (Cerulean Ping Protocol packets, id 3104)
  * RViz with config/coastal_survey.rviz (rviz:=true, default)

See vrx_gz/worlds/coastal/README.md.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import vrx_gz.launch
from vrx_gz.model import Model


def launch(context, *args, **kwargs):
    cfg = lambda k: LaunchConfiguration(k).perform(context)  # noqa: E731
    share = get_package_share_directory('vrx_gz')
    world = os.path.splitext(cfg('world'))[0]
    headless = cfg('headless').lower() == 'true'
    paused = cfg('paused').lower() == 'true'
    sim_mode = cfg('sim_mode')

    with open(os.path.join(share, 'config', 'coastal_spawn_poses.yaml')) as f:
        pose = list(yaml.safe_load(f).get(world, [0, 0, 0, 0, 0, 0]))
    for i, k in enumerate(['x', 'y', 'z', 'R', 'P', 'yaw']):
        if cfg(k) != '':
            pose[i] = float(cfg(k))

    urdf = cfg('urdf') or os.path.join(
        get_package_share_directory('wamv_gazebo'), 'urdf', 'wamv_survey.urdf.xacro')
    robot = Model('wamv', 'wam-v', pose)
    robot.set_urdf(urdf)

    actions = []
    actions += vrx_gz.launch.simulation(world, headless, paused,
                                        cfg('extra_gz_args'))
    actions += vrx_gz.launch.spawn(sim_mode, world, [robot], 'wamv')
    if sim_mode in ('full', 'bridge'):
        actions += vrx_gz.launch.competition_bridges(world, False)
        actions.append(Node(
            package='ros_gz_bridge', executable='parameter_bridge',
            name='coastal_survey_bridge', output='screen',
            arguments=['/model/wamv/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry'],
            remappings=[('/model/wamv/odometry', '/wamv/ground_truth/odometry')]))
        actions.append(Node(
            package='tf2_ros', executable='static_transform_publisher',
            name='wamv_frame_join', output='log',
            arguments=['--frame-id', 'wamv/base_link',
                       '--child-frame-id', 'wamv/wamv/base_link'],
            parameters=[{'use_sim_time': True}]))
        actions.append(Node(
            package='vrx_gz', executable='lidar_water_filter.py',
            name='lidar_water_filter', output='screen',
            parameters=[{'use_sim_time': True, 'water_level': 0.0, 'margin': 0.0}]))
        fresh = world.startswith('claytor_lake')   # reservoir: fresh water
        actions.append(Node(
            package='vrx_gz', executable='omniscan3d_sim.py',
            name='omniscan3d_sim', output='screen',
            parameters=[{'use_sim_time': True,
                         'sound_speed_mps': 1480.0 if fresh else 1500.0,
                         'absorption_db_per_m': 0.05 if fresh else 0.12}]))
        if cfg('rviz').lower() == 'true':
            actions.append(Node(
                package='rviz2', executable='rviz2', name='rviz2', output='log',
                arguments=['-d', cfg('rviz_config') or
                           os.path.join(share, 'config', 'coastal_survey.rviz')],
                parameters=[{'use_sim_time': True}]))
    return actions


def generate_launch_description():
    args = [
        DeclareLaunchArgument('world', default_value='nbs_surf_epoch0',
                              description='claytor_lake_calm | cliff_coast_epoch{0,1,2} | '
                                          'nbs_surf_epoch{0,1,2} | ocean_view_norfolk'),
        DeclareLaunchArgument('rviz', default_value='true',
                              description='start RViz with the survey layout'),
        DeclareLaunchArgument('rviz_config', default_value='',
                              description='RViz config (default: config/coastal_survey.rviz)'),
        DeclareLaunchArgument('sim_mode', default_value='full',
                              description='full | sim | bridge'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('paused', default_value='false'),
        DeclareLaunchArgument('extra_gz_args', default_value=''),
        DeclareLaunchArgument('urdf', default_value='',
                              description='robot xacro (default: survey WAM-V)'),
    ]
    for k in ['x', 'y', 'z', 'R', 'P', 'yaw']:
        args.append(DeclareLaunchArgument(
            k, default_value='', description=f'spawn {k} override'))
    return LaunchDescription(args + [OpaqueFunction(function=launch)])
