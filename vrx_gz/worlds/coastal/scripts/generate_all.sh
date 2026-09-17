#!/bin/bash
# Generate every coastal simulation asset: terrain meshes and textures
# (vrx_gz/models/*_terrain*/), world SDFs (vrx_gz/worlds/*.sdf), ground truth
# and material maps (vrx_gz/worlds/coastal/ground_truth/).
#
# The meshes and ground truth are NOT tracked in git (they are ~700 MB of
# generated data), so run this once after cloning, and again after changing a
# generator. Then rebuild so the assets are installed:
#
#   ./generate_all.sh
#   cd ~/workspace/simulator && colcon build --merge-install --packages-select vrx_gz
#
# Options:
#   --skip-ocean-view   skip the real NOAA CUDEM world (needs rasterio and
#                       network access)
#   PYTHON=<python>     interpreter to use (default: python3)
#
# Requirements: see requirements.txt (numpy, scipy, pillow, matplotlib,
# pyyaml, and rasterio for ocean_view_norfolk / GeoTIFF export).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
SKIP_OV=0
for a in "$@"; do
  case "$a" in
    --skip-ocean-view) SKIP_OV=1 ;;
    *) echo "unknown option: $a"; exit 2 ;;
  esac
done

echo "== checking requirements ($PY)"
$PY - <<'EOF'
import importlib.util
import sys
missing = [m for m, p in [('numpy', 'numpy'), ('scipy', 'scipy'), ('PIL', 'pillow'),
                          ('matplotlib', 'matplotlib'), ('yaml', 'pyyaml')]
           if not importlib.util.find_spec(m)]
if missing:
    sys.exit('missing python modules: ' + ' '.join(missing) +
             '\n  pip install -r requirements.txt')
EOF

echo "== synthetic worlds (claytor_lake_calm, nbs_surf_epoch0-2)"
$PY gen_synthetic.py
echo "== cliff coast worlds (cliff_coast_epoch0-2)"
$PY gen_cliff_coast.py
if [ "$SKIP_OV" -eq 0 ]; then
  if $PY -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("rasterio") else 1)'; then
    echo "== real topobathy world (ocean_view_norfolk, NOAA CUDEM download)"
    $PY gen_ocean_view.py
  else
    echo "!! rasterio not installed: skipping ocean_view_norfolk"
    echo "   (pip install rasterio, then: $PY gen_ocean_view.py)"
  fi
else
  echo "== skipping ocean_view_norfolk (--skip-ocean-view)"
fi
echo "== material maps (LiDAR intensity / sonar backscatter)"
$PY gen_material_maps.py $([ "$SKIP_OV" -eq 1 ] && echo "--worlds claytor_lake_calm nbs_surf_epoch0 nbs_surf_epoch1 nbs_surf_epoch2 cliff_coast_epoch0 cliff_coast_epoch1 cliff_coast_epoch2" || true)

echo
echo "done. Now rebuild so the assets are installed:"
echo "  cd ~/workspace/simulator && colcon build --merge-install --packages-select vrx_gz"
