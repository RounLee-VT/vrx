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
| `vrx_gz/models/*_terrain*/` | terrain meshes and textures (**generated, not in git**) |
| `vrx_gz/launch/coastal_survey.launch.py` | launch: world + survey WAM-V + bridges + TF join + LiDAR/sonar post-processing + RViz |
| `vrx_gz/config/coastal_survey.rviz` | RViz layout |
| `vrx_gz/scripts/lidar_sim.py` | LiDAR: removes through-water and self returns; material-dependent intensity, detection limit, material label |
| `vrx_gz/src/vrx_gz/coastal_materials.py`, `vrx_gz/config/coastal_materials.yaml` | material map lookup and material property table (LiDAR reflectivity, sonar backscatter) |
| `vrx_gz/scripts/omniscan3d_sim.py` | Omniscan 3D output: along-track beam merge, TOF, power, `OS3D_POINT_SET` packets |
| `vrx_gz/config/coastal_spawn_poses.yaml` | default spawn pose per world |
| `vrx_gz/config/nbs_survey_transects.yaml` | repeat-survey lines |
| `vrx_urdf/wamv_gazebo/urdf/wamv_survey.urdf.xacro` | survey WAM-V |
| `vrx_gz/worlds/coastal/scripts/` | generators (`generate_all.sh`, `requirements.txt`) |
| `vrx_gz/worlds/coastal/ground_truth/` | DEMs, masks, targets, change maps, material maps (**generated, not in git**) |

## Run

The terrain meshes and ground truth (~700 MB) are generated, not tracked in
git, so generate them once after cloning:

```bash
cd ~/workspace/simulator/vrx/vrx_gz/worlds/coastal/scripts
pip install -r requirements.txt     # numpy scipy pillow matplotlib pyyaml rasterio
./generate_all.sh                   # ~40 s; --skip-ocean-view to skip the NOAA download

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
| Omniscan 3D 450 SS proxy | (0.40, 0.00, −0.30), ~0.4 m below waterline | nadir; sensor frame x = down, y = starboard | `/wamv/sensors/sonars/omniscan3d/points` (x y z angle tof pwr pt_type), `.../os3d_point_set` (raw packets), plus a `material` label field; raw ray geometry: `/wamv/sensors/lidars/omniscan3d_sensor/points` |
| IMU 200 Hz | (0.30, −0.20, 1.30) | aligned | `/wamv/sensors/imu/imu/data` |
| GPS (NavSat) | (−0.85, 0.00, 1.30) | aligned | `/wamv/sensors/gps/gps/fix` |
| Front camera | (0.75, 0.30, ~1.5) | pitched 15° down | `/wamv/sensors/cameras/front_camera_sensor/image_raw` |
| Ground-truth odometry 50 Hz | base_link | – | `/wamv/ground_truth/odometry` (`world` → `wamv/base_link`) |

**TF:** the launch file joins the VRX pose tree (`world → wamv/base_link`) and
the robot/sensor tree (`wamv/wamv/base_link → ...`) with an identity static
transform, so every sensor frame resolves in `world`.

**LiDAR post-processing (`lidar_sim.py`, topic `points_filtered`).** Use this topic rather than the raw cloud.

* **Through-water returns:** VRX GPU LiDARs ignore the wave visual (visibility
  mask 7), so raw beams pass through the water and return the seabed; in the
  cliff world this is 55–75 % of the points. A real near-IR LiDAR gets no such
  returns. The node removes points below the water level (world z < 0), using
  the TF at the scan time.
* **Own vessel:** hits inside a crop box around the hull are removed.
* **Intensity:** Gazebo's intensity channel is always 0 here: it only
  supports one `laser_retro` value per visual, and each terrain is a single
  mesh. The node therefore synthesises
  `intensity = 100 · ρ(material) · cos θᵢ · (1 + 5 % noise)`, on a
  Velodyne-style calibrated scale where 0–100 covers diffuse targets.
  * ρ is the ~905 nm reflectivity of the point's material, looked up in the
    world's material map.
  * θᵢ is the incidence angle, from the scan-grid surface normal, falling
    back to the ground-truth DEM normal for grazing terrain hits.
* **Detection limit:** returns with `ρ·cos θᵢ·(100 m / r)² < 0.10` are
  dropped, i.e. a 10 % target is detected out to 100 m. Dark or grazing
  surfaces therefore drop out at long range.
* **Output fields:** `x y z intensity signal_db ring material`.
  `signal_db = 10 log10(ρ cos θᵢ / r²)` is the range-dependent return
  strength, and `material` is a ground-truth label.

**Material maps.** `vrx_gz/worlds/coastal/scripts/gen_material_maps.py`
builds `ground_truth/<world>_materials.npz/.png` from the exported masks and
elevation rules; it does not touch the meshes.

* Materials: sand (dry, wet, seabed), mud, grass, forest, marsh, bare soil,
  rock, wet rock, algae, oyster reef, talus, concrete, and `object` for
  anything standing clear of the terrain (targets, pilings).
* The cliff world also classifies the 3D cliff face and sea stacks by height:
  rock, wet rock in the splash zone, algae in the intertidal band.
* Per-material LiDAR reflectivity and sonar Lambert μ are in
  `config/coastal_materials.yaml`. They are representative literature-range
  values and must be tuned against real sensor data.

**Omniscan 3D 450 SS proxy.** Specs are from [Cerulean docs](https://docs.ceruleansonar.com/c/omniscan3d/technical-details/specifications).

| Spec | Real device | Simulation |
|---|---|---|
| Cross-track TX beam | 90° (−10 dB) | 90° fan (±45°) |
| Angle resolution | < 1° (angle of arrival) | 181 rays, 0.5° |
| Along-track beam | 0.8° | 5 rays across ±0.4°, merged per beam with a −3 dB-edge beam-pattern weighting (`along_track_mode:=first` keeps the leading edge) |
| Max range (3D) | 100 m slant | 100 m |
| Range resolution | up to range/1000 | 0.02 m quantisation + 0.02 m Gaussian noise, applied after beam merging |
| Ping rate | ≤ 20 Hz (range limited) | 20 Hz |
| Angle convention | positive to starboard | same (sensor y = starboard) |
| Point data | angle, time of flight, power, pt_type (`OS3D_POINT_SET`) | same fields in PointCloud2, plus byte-exact `OS3D_POINT_SET` packets (id 3104, 80-byte header, 16-byte points, 'BR' framing + checksum) |
| Mounting | 30–45° down, side-looking; **straight down not supported** by the real unit | nadir (per current plan); `sonar_tilt_deg` xacro arg for side-looking |

`vrx_gz/scripts/omniscan3d_sim.py` turns the ideal ray geometry into sonar
output:

* `tof = 2 r / c` (two-way), with `c` = 1500 m/s (sea) or 1480 m/s (Claytor Lake).
* `pwr` (relative dB) from the sonar equation `SL − 2·TL + BS + B_tx + speckle`:
  * `TL = 20 log10 r + α r`, with α = 0.12 dB/m (sea) or 0.05 dB/m (fresh water), both approximate.
  * `BS = μ(material) + 10 log10 cos²θᵢ` (Lambert's law). μ comes from the world's material map, e.g. mud −36, sand −27, rock −18, talus −16 dB. θᵢ comes from the true local surface normal of the ray grid.
  * `B_tx = −10 dB · (angle/45°)²`.
  * Rayleigh speckle.
* Echoes below the −110 dB noise floor are dropped. The per-ping `pwr_threshold_high/med/low` are floor + 20/12/6 dB.
* A decoder, `decode_os3d_point_set()`, is included in the same file.

Validated in simulation:
* TOF is exact (≤ 5e-10 s).
* The nadir `pwr` residual has mean −2.51 dB and std 5.59 dB, matching Rayleigh speckle theory (−2.51 / 5.57).
* Incidence-angle estimation is exact on synthetic planes sloped up to 40° / 30°.
* Depth against the GT DEM has mean 0.001 m and std 0.019 m.
* Packet size and content match the point cloud at 20 Hz.

Assumptions (not in the vendor docs): `tof` is two-way; the absolute `pwr`
scale, thresholds and backscatter parameters are simulation conventions that
need calibration against real data.

Not modelled: interferometric angle noise, auto range and the 5 % minimum
start range, sound-speed refraction, multipath, water-column targets,
sub-material texture (grain size, vegetation) within a material class, and the built-in pitch/roll IMU (the vessel IMU
is used instead). Gazebo has no acoustic sensor, so the geometry comes from a
GPU ray fan that skips the water surface.
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
* **Sonar:** Omniscan3D points kept for 120 s, so the swath map builds up as the boat moves. Coloured by depth; a second display coloured by `pwr` is off by default.
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

## Generate / modify

`generate_all.sh` runs all four generators; the output is deterministic, so
regenerating gives byte-identical files. Run them individually to change a
world:

```bash
cd ~/workspace/simulator/vrx/vrx_gz/worlds/coastal/scripts
python3 gen_synthetic.py [--sea-state calm|light|moderate|rough] [--only lake|nbs]
python3 gen_cliff_coast.py [--sea-state moderate] [--epochs 0 1 2]
python3 gen_ocean_view.py [--water-level 0.4] [--sea-state rough] [--res 3.0] \
                          [--bounds W S E N] [--name my_site]
python3 gen_material_maps.py [--worlds ...]   # after any terrain change
```

Rebuild `vrx_gz` afterwards so the assets are installed.

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
