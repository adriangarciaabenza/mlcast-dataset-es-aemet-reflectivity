#!/bin/bash
#SBATCH --job-name=merge_zarr_v3_full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=/perm/pred/std/ML/MLCAST/mlcast-dataset-ES-AEMET-reflectivity/logs/log_merge_zarr_%j.out
#SBATCH --error=/perm/pred/std/ML/MLCAST/mlcast-dataset-ES-AEMET-reflectivity/logs/log_merge_zarr_%j.err

set -euo pipefail

# =========================================================
# LOAD ENV
# =========================================================
set +u
source /etc/profile
set -u

PROJECT_DIR=/perm/pred/std/ML/MLCAST/mlcast-dataset-ES-AEMET-reflectivity

cd "${PROJECT_DIR}"

# Activar entorno
source "${PROJECT_DIR}/.venv/bin/activate"

# =========================================================
# ENV VARIABLES (importantes para rendimiento Dask)
# =========================================================
export PYTHONUNBUFFERED=1

# evita sobrecarga de threads (MUY importante en HPC)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Dask control
export DASK_NUM_WORKERS=8

# =========================================================
# DEBUG INFO
# =========================================================
echo "=== JOB START ==="
date
hostname
echo "CPUS: $SLURM_CPUS_PER_TASK"
echo "MEM: $SLURM_MEM_PER_NODE"
echo "PWD: $(pwd)"

# =========================================================
# RUN MERGE
# =========================================================

python scripts/merge_monthly_zarrs_2.py

# =========================================================
# END
# =========================================================
echo "=== JOB END ==="
date