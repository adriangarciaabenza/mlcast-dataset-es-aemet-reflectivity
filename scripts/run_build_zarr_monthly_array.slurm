#!/bin/bash
#SBATCH --job-name=build_zarr_v3_monthly
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=13-24%4
#SBATCH --output=/perm/pred/std/ML/MLCAST/mlcast-dataset-ES-AEMET-reflectivity/logs/log_build_zarr_monthly_%A_%a.out
#SBATCH --error=/perm/pred/std/ML/MLCAST/mlcast-dataset-ES-AEMET-reflectivity/logs/log_build_zarr_monthly_%A_%a.err

set -euo pipefail

set +u
source /etc/profile
set -u

PROJECT_DIR=/perm/pred/std/ML/MLCAST/mlcast-dataset-ES-AEMET-reflectivity
CONFIG_LIST=${PROJECT_DIR}/configs/monthly_v3/config_list.txt

# ajusta esto a tu entorno real
source /perm/pred/std/ML/MLCAST/mlcast-dataset-ES-AEMET-reflectivity/.venv/bin/activate

export PYTHONUNBUFFERED=1
export AEMET_API_KEY="Need to obtain API KEY from http://bigdata.aemet.es/bigdata/altaUsuario?"
CONFIG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${CONFIG_LIST}")

echo "=== JOB START ==="
date
hostname
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Config: ${CONFIG}"

cd "${PROJECT_DIR}"

python scripts/build_zarr.py --config "${CONFIG}"

echo "=== JOB END ==="
date