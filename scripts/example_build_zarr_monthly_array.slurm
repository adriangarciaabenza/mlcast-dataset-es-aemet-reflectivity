#!/bin/bash
#SBATCH --job-name=build_zarr_v3_monthly
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-12%4
#SBATCH --output=logs/build_zarr_monthly_%A_%a.out
#SBATCH --error=logs/build_zarr_monthly_%A_%a.err

set -euo pipefail

# This is an example SLURM array script.
# Adapt paths, modules, environment activation, array range, and resources to your HPC system.

source /etc/profile

# -------------------------------------------------------------------------
# User configuration
# -------------------------------------------------------------------------
PROJECT_DIR="/path/to/mlcast-dataset-ES-AEMET-reflectivity"
VENV_DIR="${PROJECT_DIR}/.venv"
CONFIG_LIST="${PROJECT_DIR}/configs/monthly_v3/config_list.txt"

export PYTHONUNBUFFERED=1
export AEMET_API_KEY="${AEMET_API_KEY:?Please export AEMET_API_KEY before running this script.}"

source "${VENV_DIR}/bin/activate"

CONFIG_PATH="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${CONFIG_LIST}")"

if [[ -z "${CONFIG_PATH}" ]]; then
    echo "No config file was found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "=== JOB START ==="
date
hostname
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Config path: ${CONFIG_PATH}"

cd "${PROJECT_DIR}"

python scripts/build_zarr.py --config "${CONFIG_PATH}"

echo "=== JOB END ==="
date