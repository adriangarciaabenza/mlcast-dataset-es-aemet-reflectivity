#!/bin/bash
#SBATCH --job-name=merge_zarr_v3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/perm/pred/std/ML/MLCAST/mlcast-dataset-ES-AEMET-reflectivity/logs/log_merge_zarr_v3_%j.out
#SBATCH --error=/perm/pred/std/ML/MLCAST/mlcast-dataset-ES-AEMET-reflectivity/logs/log_merge_zarr_v3_%j.err

set -euo pipefail

source /etc/profile
source /perm/pred/std/ML/MLCAST/.venv/bin/activate

export PYTHONUNBUFFERED=1

PROJECT_DIR=/perm/pred/std/ML/MLCAST/mlcast-dataset-ES-AEMET-reflectivity
cd "${PROJECT_DIR}"

python scripts/merge_monthly_zarr_v3.py