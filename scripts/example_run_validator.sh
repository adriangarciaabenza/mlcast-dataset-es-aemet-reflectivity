#!/bin/bash
set -euo pipefail

# =============================================================================
# Example script to install and run the MLCAST dataset validator
# =============================================================================
# This script:
#   1. Optionally loads HPC modules
#   2. Creates a Python environment (uv)
#   3. Installs required dependencies (GDAL, Cartopy, validator)
#   4. Runs the validator on a Zarr dataset
#
# Adapt paths and module names to your system.
# =============================================================================


echo "=== START VALIDATOR SETUP ==="
date
hostname


# =============================================================================
# OPTIONAL: Load HPC modules (adapt to your system)
# =============================================================================
if command -v module &> /dev/null; then
    echo "=== LOADING MODULES ==="
    module purge

    module load gnu/8.3.1
    module load openmpi/4.0.5.2
    module load proj/9.0.0
    module load geos
    module load gdal/3.10.2
fi


# =============================================================================
# USER CONFIGURATION
# =============================================================================
ENV_DIR="${ENV_DIR:-./mlcast_validator_env}"
DATASET_PATH="${DATASET_PATH:-./example_dataset.zarr}"


# =============================================================================
# CHECK DATASET
# =============================================================================
if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "ERROR: Dataset path does not exist: ${DATASET_PATH}"
    echo "Please set DATASET_PATH to a valid Zarr dataset."
    exit 1
fi


# =============================================================================
# PREPARE PYTHON ENVIRONMENT
# =============================================================================
echo "=== PREPARING ENVIRONMENT ==="

# Check uv
if ! command -v uv &> /dev/null; then
    echo "ERROR: 'uv' is not installed."
    echo "Install it from: https://github.com/astral-sh/uv"
    exit 1
fi

# Create environment if it does not exist
if [[ ! -d "${ENV_DIR}" ]]; then
    echo "Creating environment at: ${ENV_DIR}"
    uv venv --python 3.11 "${ENV_DIR}"
fi

# Activate environment
source "${ENV_DIR}/bin/activate"

echo "Python executable:"
which python
python --version


# =============================================================================
# GDAL CONFIGURATION
# =============================================================================
if command -v gdal-config &> /dev/null; then
    export GDAL_CONFIG="$(which gdal-config)"
    echo "Using GDAL_CONFIG=${GDAL_CONFIG}"
else
    echo "WARNING: gdal-config not found in PATH"
fi


# =============================================================================
# INSTALL DEPENDENCIES
# =============================================================================
echo "=== INSTALLING DEPENDENCIES ==="

# Match GDAL version with system if possible
if command -v gdal-config &> /dev/null; then
    GDAL_VERSION=$(gdal-config --version)
    echo "Detected GDAL version: ${GDAL_VERSION}"

    uv pip install --upgrade --force-reinstall "GDAL[numpy]==${GDAL_VERSION}.*"
else
    echo "Installing generic GDAL version (may fail depending on system)"
    uv pip install --upgrade GDAL[numpy]
fi

uv pip install --upgrade --force-reinstall cartopy
uv pip install --upgrade --force-reinstall rioxarray
uv pip install --upgrade --force-reinstall \
    "git+https://github.com/mlcast-community/mlcast-dataset-validator"


# =============================================================================
# QUICK CHECKS
# =============================================================================
echo "=== CHECKING INSTALLATION ==="

python -c "from osgeo import gdal; print('GDAL OK:', gdal.VersionInfo())"
python -c "import cartopy; print('Cartopy OK:', cartopy.__version__)"
python -c "import mlcast_dataset_validator; print('Validator OK')"


# =============================================================================
# RUN VALIDATOR
# =============================================================================
echo "=== RUNNING VALIDATOR ==="

mlcast.validate_dataset \
    source_data \
    radar_precipitation \
    "${DATASET_PATH}"


echo "=== VALIDATION FINISHED ==="
date