# # 1) Resetear módulos
module purge

# # 2) Cargar stack nativo necesario
module load gnu/8.3.1
module load openmpi/4.0.5.2
module load gdal/3.10.2

# # 3) Verificar y exportar gdal-config
which gdal-config
gdal-config --version
export GDAL_CONFIG=$(which gdal-config)

# # 4) Crear un entorno uv con Python 3.11 aparte
uv venv --python 3.11 /lustre/utmp/std/uv_envs/new_mlcast_validator_py311

# # 5) Activarlo
source /lustre/utmp/std/uv_envs/new_mlcast_validator_py311/bin/activate

# # 6) Instalar GDAL compatible con la librería nativa del sistema
uv pip install --force-reinstall "GDAL[numpy]==3.10.2.*"

# # 7) Instalar Cartopy
uv pip install --force-reinstall cartopy
uv pip install --force-reinstall rioxarray

# # 8) Instalar el validator desde GitHub
uv pip install --force-reinstall "git+https://github.com/mlcast-community/mlcast-dataset-validator"

# # 9) Comprobaciones rápidas
python -c "from osgeo import gdal; print(gdal.VersionInfo())"
python -c "import cartopy; print(cartopy.__version__)"
python -c "import mlcast_dataset_validator; print('validator OK')"

deactivate
# # 10) Ejecutar el validator
# mlcast.validate_dataset source_data radar_precipitation ./radar_test_geozarr_ZAR_2024_1001.zarr/

#!/bin/bash

module purge
module load gnu/8.3.1
module load openmpi/4.0.5.2
module load proj/9.0.0
module load geos
module load gdal/3.10.2

export GDAL_CONFIG=$(which gdal-config)
export UV_LINK_MODE=copy

source /lustre/utmp/std/uv_envs/new_mlcast_validator_py311/bin/activate

#mlcast.validate_dataset source_data radar_precipitation ./radar_test_geozarr_ZAR_2024_1001.zarr/
#mlcast.validate_dataset source_data radar_precipitation ./ES-AEMET-radar_reflectivity-ppi_ZAR.zarr/
#mlcast.validate_dataset source_data radar_precipitation ./test_radar_small_v3.zarr
#mlcast.validate_dataset source_data radar_precipitation /lustre/utmp/std/MLCAST_radar_data/ES-AEMET-radar_reflectivity-ppi_ZAR_v3.zarr
#mlcast.validate_dataset source_data radar_precipitation /perm/pred/std/ML/MLCAST/mlcast-dataset-ES-AEMET-reflectivity/ES-AEMET-radar_reflectivity-ppi_ZAR_2020-2024_v2.zarr
#mlcast.validate_dataset source_data radar_precipitation /lustre/utmp/std/MLCAST_radar_data/ES-AEMET-radar_reflectivity-ppi_ZAR_2020-2024_v3.zarr
#mlcast.validate_dataset source_data radar_precipitation /lustre/utmp/std/MLCAST_radar_data/ES-AEMET-rainfall_rate-sri_ZAR_2020-2024_v1_kg.zarr
mlcast.validate_dataset source_data radar_precipitation /lustre/utmp/std/MLCAST_radar_data/ES-AEMET-rainfall_rate-sri_ZAR_2020-2024_v1_converted.zarr
