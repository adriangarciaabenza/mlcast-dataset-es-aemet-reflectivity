#!/bin/bash
#SBATCH --job-name=build_zarr_v3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/build_zarr_%j.out
#SBATCH --error=logs/build_zarr_%j.err

set -euo pipefail

# This is an example SLURM script.
# Adapt paths, modules, environment activation, and resources to your HPC system.

echo "=== JOB START ==="
date
hostname

# Optional: load your site profile if needed
source /etc/profile

# -------------------------------------------------------------------------
# User configuration
# -------------------------------------------------------------------------
PROJECT_DIR="/path/to/mlcast-dataset-ES-AEMET-reflectivity"
VENV_DIR="${PROJECT_DIR}/.venv"
CONFIG_PATH="${PROJECT_DIR}/configs/example_build_zarr.yaml"

export PYTHONUNBUFFERED=1
export AEMET_API_KEY="${AEMET_API_KEY:?Please export AEMET_API_KEY before running this script.}"

# -------------------------------------------------------------------------
# Environment activation
# -------------------------------------------------------------------------
source "${VENV_DIR}/bin/activate"

echo "Python executable:"
which python
python --version

# -------------------------------------------------------------------------
# Run
# -------------------------------------------------------------------------
cd "${PROJECT_DIR}"

echo "=== CONFIG ==="
cat "${CONFIG_PATH}"

echo "=== RUNNING BUILD SCRIPT ==="
python scripts/build_zarr.py --config "${CONFIG_PATH}"

EXIT_CODE=$?

echo "=== JOB END ==="
date
echo "Exit code: ${EXIT_CODE}"

exit "${EXIT_CODE}"