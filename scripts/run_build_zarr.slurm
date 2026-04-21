#!/bin/bash
#SBATCH --job-name=build_zarr_v3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/perm/pred/std/ML/MLCAST/mlcast-dataset-ES-AEMET-reflectivity/logs/log_build_zarr_%j.out
#SBATCH --error=/perm/pred/std/ML/MLCAST/mlcast-dataset-ES-AEMET-reflectivity/logs/log_build_zarr_%j.err

# =============================================================================
# ENTORNO
# =============================================================================

echo "=== JOB START ==="
date
hostname

# Cargar entorno (IMPORTANTE en tu sistema)
source /etc/profile

# Activa tu entorno
source /perm/pred/std/ML/MLCAST/mlcast-dataset-ES-AEMET-reflectivity/.venv/bin/activate

echo "Python ejecutándose desde:"
which python
python --version

# =============================================================================
# VARIABLES
# =============================================================================

PROJECT_DIR=/perm/pred/std/ML/MLCAST/mlcast-dataset-ES-AEMET-reflectivity
CONFIG=${PROJECT_DIR}/configs/AEMET-PPI_v3.yaml

# API KEY AEMET
export AEMET_API_KEY="eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhZ2FyY2lhYUBhZW1ldC5lcyIsImp0aSI6ImM4NjRmMzc1LWY4OTgtNDlkNy1hZDFmLWQ3YTcwNDQ2ODRhYSIsImlzcyI6IkFFTUVUIiwiaWF0IjoxNjc4MjY4NzcwLCJ1c2VySWQiOiJjODY0ZjM3NS1mODk4LTQ5ZDctYWQxZi1kN2E3MDQ0Njg0YWEiLCJyb2xlIjoiIn0._f4rSIByCWg1CsxfdZFnxiextF6xYKTBMp9l5TjHqp4"

# Para logs en tiempo real
export PYTHONUNBUFFERED=1

# =============================================================================
# EJECUCIÓN
# =============================================================================

cd ${PROJECT_DIR}

echo "=== CONFIG ==="
cat ${CONFIG}

echo "=== RUN BUILD_ZARR ==="

python scripts/build_zarr.py \
    --config ${CONFIG}

EXIT_CODE=$?

echo "=== JOB END ==="
date
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}