# Coastal survey worlds (vrx_gz)

Worlds for simulation-based pre-validation of the VT side of the 4-VA project
*AI-Powered Autonomous Coastal Monitoring and 3D Change Detection* (ASV survey,
LiDAR + sonar mapping, wave-disturbance compensation). Built on VRX
(Gazebo Harmonic, ROS 2 Jazzy).

| World | What it is for | Terrain | Sea state |
|---|---|---|---|
| `claytor_lake_calm` | Milestone 1 still-water ground-truth test (mapping accuracy, sensor extrinsics) | synthetic reservoir cove in hill country: 40–55 m wooded hills, ~25 m rock bluff, calibration targets, 3 km coarse surrounding valley | calm (no waves) |
| `cliff_coast_epoch0/1/2` | cliff erosion / 3D change detection (cf. proposal Fig. 1) | synthetic ~30 m layered sea cliff (true 3D mesh) with wave-cut notch, sea caves, gullies, sea stacks, jointed rock platform, pocket beach | moderate |
| `nbs_surf_epoch0/1/2` | repeat surveys (Oct 26 / Jan 27 / Mar 27) for change detection and NBS evaluation | synthetic Chesapeake-like shoreline: living shoreline (marsh + oyster sills), control beach, rock breakwaters | moderate (Hs 0.5 m, Tp 3.5 s) |
| `ocean_view_norfolk` | realistic Hampton Roads site: surf zone, breakwaters, Little Creek Inlet jetties | **real** NOAA CUDEM 1/9" topobathy (~3 m) | moderate |

No off-the-shelf Gazebo world covers a surf zone or NBS site (DAVE/UUV Simulator
only have deep seabed heightmaps), so the worlds are generated: one from public
NOAA data and the rest procedurally, with exact ground truth.

## Layout

| Path (in the vrx repo) | Content |
|---|---|
| `vrx_gz/worlds/{claytor_lake_calm,cliff_coast_epoch0-2,nbs_surf_epoch0-2,ocean_view_norfolk}.sdf` | worlds |
| `vrx_gz/models/*_terrain/` | terrain meshes and textures (generated) |
| `vrx_gz/launch/coastal_survey.launch.py` | launch: world + survey WAM-V + bridges + TF join + LiDAR water filter + RViz |
| `vrx_gz/config/coastal_survey.rviz` | RViz layout |
| `vrx_gz/scripts/lidar_water_filter.py` | removes through-water and self LiDAR returns |
| `vrx_gz/config/coastal_spawn_poses.yaml` | default spawn pose per world |
| `vrx_gz/config/nbs_survey_transects.yaml` | repeat-survey lines |
| `vrx_urdf/wamv_gazebo/urdf/wamv_survey.urdf.xacro` | survey WAM-V |
| `vrx_gz/worlds/coastal/scripts/` | generators |
| `vrx_gz/worlds/coastal/ground_truth/` | DEMs, masks, targets, change maps |

## Run

```bash
cd ~/workspace/simulator
colcon build --merge-install --packages-select vrx_gz wamv_gazebo
source install/setup.bash

ros2 launch vrx_gz coastal_survey.launch.py world:=nbs_surf_epoch0
ros2 launch vrx_gz coastal_survey.launch.py world:=cliff_coast_epoch0
ros2 launch vrx_gz coastal_survey.launch.py world:=ocean_view_norfolk
ros2 launch vrx_gz coastal_survey.launch.py world:=claytor_lake_calm headless:=true rviz:=false
# options: rviz:=true|false (default true), headless:=true, spawn x:= y:= z:= R:= P:= yaw:=
#          (spawn defaults: config/coastal_spawn_poses.yaml)

python3 ~/workspace/simulator/vrx/controller.py   # keyboard teleop (another terminal)
```

The worlds also load with `competition.launch.py world:=nbs_surf_epoch0`, but
that launcher spawns at VRX's Sydney coordinates and uses the default WAM-V, so
use `coastal_survey.launch.py`.

### Survey WAM-V (`vrx_urdf/wamv_gazebo/urdf/wamv_survey.urdf.xacro`)

Extrinsics are given in `wamv/base_link` (x forward, y port, z up). z = 0 is the
pontoon bottom, the deck is at ~1.3 m, and the still-water draft is ~0.1 m.
All values were verified in the simulation: URDF, Gazebo sensor pose and TF
agree, IMU attitude matches ground truth to 0.03°, and the sonar has zero
vertical bias against the GT DEM (σ 0.02 m).

| Sensor | Position (m) | Orientation | ROS topic (frame) |
|---|---|---|---|
| 3D LiDAR 16-beam ±15°, 10 Hz | (0.70, 0.00, 2.30) | level (rpy 0 0 0) | `/wamv/sensors/lidars/lidar_wamv_sensor/points` (raw), `.../points_filtered` (above water, no self-hits) |
| Omniscan 3D 450 SS proxy | (0.40, 0.00, −0.30), ~0.4 m below waterline | nadir; sensor frame x = down, y = starboard | `/wamv/sensors/lidars/omniscan3d_sensor/points`, `.../scan` |
| IMU 200 Hz | (0.30, −0.20, 1.30) | aligned | `/wamv/sensors/imu/imu/data` |
| GPS (NavSat) | (−0.85, 0.00, 1.30) | aligned | `/wamv/sensors/gps/gps/fix` |
| Front camera | (0.75, 0.30, ~1.5) | pitched 15° down | `/wamv/sensors/cameras/front_camera_sensor/image_raw` |
| Ground-truth odometry 50 Hz | base_link | – | `/wamv/ground_truth/odometry` (`world` → `wamv/base_link`) |

**TF:** the launch file joins the VRX pose tree (`world → wamv/base_link`) and
the robot/sensor tree (`wamv/wamv/base_link → ...`) with an identity static
transform, so every sensor frame resolves in `world`.

**LiDAR through water:** VRX GPU LiDARs ignore the wave visual (visibility mask
7), so raw beams pass through the water and return the seabed; in the cliff
world this is 55–75 % of the points. A real near-IR LiDAR gets no such
returns. `lidar_water_filter.py` removes points below the water level (world
z < 0), using the TF at the scan time. It also removes hits on the own
vessel. Use `points_filtered` for above-water mapping.

**Omniscan 3D 450 SS proxy.** Specs are from [Cerulean docs](https://docs.ceruleansonar.com/c/omniscan3d/technical-details/specifications).

| Spec | Real device | Simulation |
|---|---|---|
| Cross-track TX beam | 90° (−10 dB) | 90° fan (±45°) |
| Angle resolution | < 1° (angle of arrival) | 181 rays, 0.5° |
| Along-track beam | 0.8° | not modelled (single ray row) |
| Max range (3D) | 100 m slant | 100 m |
| Range resolution | up to range/1000 | 0.02 m quantisation + 0.02 m Gaussian noise |
| Ping rate | ≤ 20 Hz (range limited) | 20 Hz |
| Angle convention | positive to starboard | same (sensor y = starboard) |
| Output | angle, time of flight, power per point (OS3D_POINT_SET) | x/y/z points and LaserScan (range per angle); no TOF/power |
| Mounting | 30–45° down, side-looking; **straight down not supported** by the real unit | nadir (per current plan); `sonar_tilt_deg` xacro arg for side-looking |

Not modelled: interferometric angle noise, auto range and the 5 % minimum
start range, sound-speed refraction, multipath, water-column targets, and the
built-in pitch/roll IMU. Gazebo has no acoustic sensor, so this is a GPU ray
fan that skips the water surface and returns seabed geometry.
Sensor parameters are xacro args at the top of the file (`sonar_*`,
`lidar_type`). The VRX spawner passes only its fixed args, so change the
defaults there.

### RViz

`coastal_survey.launch.py` starts RViz with `config/coastal_survey.rviz` by
default. Use `rviz:=false` to disable it, or `rviz_config:=<file>` to use
another layout.

* **Fixed frame:** `world`. The orbit view follows the boat.
* **Water-level grid:** z = 0, 10 m cells.
* **WAM-V model and sensor frames:** LiDAR, sonar, IMU and GPS axes with names, to check extrinsics.
* **LiDAR (above water):** coloured by height. Raw LiDAR is a separate display, off by default.
* **Sonar:** Omniscan3D points kept for 120 s, so the swath map builds up as the boat moves.
* **Ground-truth track.**
* **Front camera image.**

**Wave limitation:** VRX waves are deep-water, spatially uniform PMS/Gerstner
waves. There is no shoaling, refraction or breaking, so the "surf zone" moves
the vessel realistically for its sea state but does not reproduce breaking
waves. The terrain collision mesh stops the boat at the beach (tested: it
grounds at ~0.5 m depth).

## Cliff coast worlds

The cliff face and sea stacks are true 3D meshes rather than heightfields,
so overhangs (wave-cut notch, caves) exist in the geometry and are visible to
LiDAR, sonar and cameras. Recesses are darkened in the texture (baked
occlusion), because the scene lighting does not darken them on its own.

A face point is `P = (x, y_c(x)) + r(x, z) * n(x)`, where `n` is the seaward
normal and `r < 0` is landward. Face retreat between epochs is exactly
`dr = r_b - r_a`.

| Epoch | Prescribed change |
|---|---|
| 0 | baseline |
| 1 | block rockfall at x 40–75 m (up to 6.7 m retreat), talus cone, notch deepening (+0.8 m), slender stack loses 7 m, pocket beach −0.6 m |
| 2 | main sea cave enlarged (+3 m deep, +2 m high), second rockfall at x −185…−165 m, talus reworked, slender stack collapses to a 3 m stump, beach partly recovers |

Ground-truth files (`ground_truth/cliff_*`):
* `cliff_coast_epoch{e}.json`: caves, stacks (centre, height, radius profile, volume), epoch parameters
* `cliff_coast_epoch{e}_ground.npz` / `_plateau.npz`: platform/seabed and clifftop DEMs with validity masks
* `cliff_coast_epoch{e}_face.npz`: `r(x, z)` face-offset map (0.5 m along-shore, 0.25 m vertical)
* `cliff_coast_epoch{e}_mesh_points.npz`: points on the exact simulated surfaces (use for cloud-to-cloud / M3C2 references)
* `cliff_change_epoch{a}_to_epoch{b}.npz/.png`, `cliff_change_summary.json`: face `dr`, ground `dz`, retreat volume, deposition, stack changes

The exact meshes are `vrx_gz/models/cliff_coast_terrain_epoch{e}/meshes/{ground,plateau,cliff}.obj`.

## Ground truth (`ground_truth/`)

All grids use the Gazebo world ENU frame (x east, y north, z up, metres, z = 0
= still water). `Z[row, col]` sits at `x = x_west + col*dx`,
`y = y_north - row*dx` (see each `*.json`). GeoTIFFs use a local transverse
Mercator CRS centred on the world's `spherical_coordinates` origin, which
matches Gazebo ENU / NavSat.

| File | Content |
|---|---|
| `<world>.npz / .tif / .json / .png` | DEM (`z`) plus class masks (`mask_oyster`, `mask_rock`, `mask_marsh`, …) and metadata |
| `claytor_lake_calm.json` → `targets` | exact poses/sizes of submerged boxes (3/6/10 m depth), sphere, 40 × 40 m flat patch at exactly −6.000 m, pier pilings and deck, shore landmarks |
| `change_epoch{a}_to_epoch{b}.npz / .png` | `dz` map, per-column shoreline position for both epochs |
| `nbs_change_summary.json` | erosion/deposition volumes and mean shoreline change for each section |
| `vrx_gz/config/nbs_survey_transects.yaml` | identical cross-shore survey lines (25 m spacing) for every epoch |

Prescribed NBS changes (see `NBS_EPOCHS` in `scripts/gen_synthetic.py`):

* **Epoch 0→1 (storm):**
  * control beach: shoreline −6 m, dune scarp
  * bar migrates 20 m offshore
  * living shoreline: marsh edge −1.5 m, lee deposition, scour at sill gaps
  * breakwaters: salients grow
* **Epoch 1→2 (partial recovery):**
  * beach partly recovers (+2.5 m)
  * bar moves 8 m back onshore
  * sill #2 damaged (crest −0.35 m, rock spread)

Fine-scale roughness is identical in every epoch, so `dz` contains only the
prescribed morphological change.

## Regenerate / modify

Generated meshes live in `vrx_gz/models/`. Rebuild `vrx_gz` after regenerating.

```bash
cd ~/workspace/simulator/vrx/vrx_gz/worlds/coastal/scripts
pip install rasterio   # needed for the real DEM and for GeoTIFF export
python3 gen_synthetic.py [--sea-state calm|light|moderate|rough] [--only lake|nbs]
python3 gen_cliff_coast.py [--sea-state moderate] [--epochs 0 1 2]
python3 gen_ocean_view.py [--water-level 0.4] [--sea-state rough] [--res 3.0] \
                          [--bounds W S E N] [--name my_site]
```

* `--water-level` is in NAVD88 metres. Use it to simulate tide stage: the DEM
  is shifted so that level becomes z = 0.
* `--bounds` can point at any area covered by the same CUDEM tile. For
  another Chesapeake Bay tile, change `TILE_URL` (tile list:
  https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/NCEI_ninth_Topobathy_2014_8483/urllist8483.txt).
* Sea-state presets live in `terrain_lib.SEA_STATES`. `gain` is derived so the
  VRX PMS wavefield gives the target Hs.
* Mesh conventions: each OBJ sub-mesh carries its own vertices and normals.
  dartsim rejects sub-meshes without normals, and an empty collision mesh
  segfaults the physics engine.

## Data source

NOAA National Centers for Environmental Information, *Continuously Updated
Digital Elevation Model (CUDEM) – 1/9 Arc-Second Resolution
Bathymetric-Topographic Tiles*, tile `ncei19_n37x00_w076x25_2019v1`
(NAD83 / NAVD88), via NOAA Digital Coast.
