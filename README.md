# VRX — Coastal Survey Simulation (Virginia Tech)

Simulation environment for the **VT side** of the 4-VA collaborative project
*AI-Powered Autonomous Coastal Monitoring and 3D Change Detection for Coastal
Resilience Using Autonomous Surface Vehicles* (VT × ODU).

This is a fork of [VRX](https://github.com/osrf/vrx) (Gazebo Harmonic + ROS 2
Jazzy) with coastal survey content added for ASV mapping and change-detection
work:

* **Coastal worlds** — a still-water reservoir (Claytor Lake test), a sea-cliff
  coast, an NBS surf zone (living shoreline, oyster sills, breakwaters), and a
  real NOAA topobathy site at Ocean View, Norfolk VA. The cliff and NBS worlds
  come in three repeat-survey epochs with exact change ground truth.
* **Survey WAM-V** — 3D LiDAR, an Omniscan 3D 450 SS sonar proxy, IMU, GPS,
  camera and ground-truth odometry, with material-dependent intensity.
* **Launch + RViz** — one command brings up a world, the vessel and a preset
  RViz layout.

The upstream VRX README is kept as [README2.md](README2.md).
Full details of the coastal content are in
[`vrx_gz/worlds/coastal/README.md`](vrx_gz/worlds/coastal/README.md).

## Requirements

* Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic (`ros-jazzy-desktop`,
  `ros-jazzy-ros-gz`, `ros-jazzy-rviz2`)
* Python: `numpy scipy pillow matplotlib pyyaml` (+ `rasterio` for the real
  NOAA world)
* A GPU is strongly recommended (the LiDAR and sonar are GPU ray sensors).

The team Docker image already contains all of this — see the `docker/`
directory of the project workspace (`docker/build.sh`, `docker-compose.yml`),
which is kept outside this repository.

## Install

```bash
# 1. clone into a colcon workspace
mkdir -p ~/workspace/simulator && cd ~/workspace/simulator
git clone git@github.com:RounLee-VT/vrx.git

# 2. generate the terrain (~700 MB of meshes, textures and ground truth).
#    These are generated, not tracked in git; run this once after cloning.
cd vrx/vrx_gz/worlds/coastal/scripts
pip install -r requirements.txt
./generate_all.sh            # ~40 s; --skip-ocean-view to skip the NOAA download

# 3. build
cd ~/workspace/simulator
colcon build --merge-install
source install/setup.bash
```

## Run

```bash
ros2 launch vrx_gz coastal_survey.launch.py world:=cliff_coast_epoch0
```

| World | Purpose |
|---|---|
| `claytor_lake_calm` | still-water ground-truth test (mapping accuracy, calibration targets) |
| `cliff_coast_epoch0/1/2` | sea-cliff erosion, 3D change detection |
| `nbs_surf_epoch0/1/2` | surf zone / nature-based solution monitoring |
| `ocean_view_norfolk` | real Hampton Roads topobathy (NOAA CUDEM) |

Options: `rviz:=false`, `headless:=true`, and spawn overrides
`x:= y:= z:= R:= P:= yaw:=` (defaults in `vrx_gz/config/coastal_spawn_poses.yaml`).

Keyboard teleop, in a second terminal:

```bash
python3 ~/workspace/simulator/vrx/controller.py
```

Main topics:

| Data | Topic |
|---|---|
| LiDAR (above water, with intensity) | `/wamv/sensors/lidars/lidar_wamv_sensor/points_filtered` |
| Sonar point cloud | `/wamv/sensors/sonars/omniscan3d/points` |
| Sonar `OS3D_POINT_SET` packets | `/wamv/sensors/sonars/omniscan3d/os3d_point_set` |
| IMU / GPS | `/wamv/sensors/imu/imu/data`, `/wamv/sensors/gps/gps/fix` |
| Ground-truth pose | `/wamv/ground_truth/odometry` |

Ground truth for each world (DEMs, class masks, change maps, material maps) is
written to `vrx_gz/worlds/coastal/ground_truth/`.

## Reference and acknowledgements

This work builds on **VRX** by Open Source Robotics Foundation and
contributors, ported to Gazebo Harmonic / ROS 2 Jazzy by
[Honu Robotics](https://honurobotics.com) with sponsorship from
[RoboNation](https://robonation.org). VRX is released under the Apache 2.0
license; please cite their publication when using the simulator:

```
@InProceedings{bingham19toward,
  Title     = {Toward Maritime Robotic Simulation in Gazebo},
  Author    = {Brian Bingham and Carlos Aguero and Michael McCarrin and
               Joseph Klamo and Joshua Malia and Kevin Allen and Tyler Lum and
               Marshall Rawson and Rumman Waqar},
  Booktitle = {Proceedings of MTS/IEEE OCEANS Conference},
  Year      = {2019},
  Address   = {Seattle, WA},
  Month     = {October}
}
```

Data and specifications used by the coastal content:

* **Terrain** — NOAA NCEI *Continuously Updated Digital Elevation Model
  (CUDEM), 1/9 arc-second Bathymetric-Topographic Tiles* (tile
  `ncei19_n37x00_w076x25_2019v1`, NAD83 / NAVD88), via
  [NOAA Digital Coast](https://coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/).
* **Sonar** — [Cerulean Sonar Omniscan 3D](https://docs.ceruleansonar.com/c/omniscan3d)
  published specifications and packet API, used to build the simulated sensor.
  Not affiliated with or endorsed by Cerulean Sonar.
* Sonar backscatter values follow typical ranges in the APL-UW *High-Frequency
  Ocean Environmental Models Handbook* (TR 9407).
