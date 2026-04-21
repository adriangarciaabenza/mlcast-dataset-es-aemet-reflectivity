from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path("/perm/pred/std/ML/MLCAST/mlcast-dataset-ES-AEMET-reflectivity")

CONFIG_DIR = PROJECT_DIR / "configs/monthly_v3"
#OUTPUT_DIR = PROJECT_DIR / "outputs/monthly_v3"
OUTPUT_DIR = Path("/lustre/utmp/std/MLCAST_radar_data/outputs/monthly_v3")

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

years = [2023, 2024]

config_paths = []

for year in years:
    for month in range(1, 13):
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1)
        else:
            end = datetime(year, month + 1, 1)

        name = f"{year}_{month:02d}"
        config_path = CONFIG_DIR / f"{name}.yaml"

        yaml_content = f"""
zarr_version: 3

workdir: "{PROJECT_DIR}/workdir/monthly_v3/{name}"
zarr_out: "{OUTPUT_DIR}/radar_{name}.zarr"

fechaini: "{start.strftime('%Y%m%dT%H%M%S')}"
fechafin: "{end.strftime('%Y%m%dT%H%M%S')}"

imagen: "PPI"
configuracion: "Z_005_240"
radar: "ZAR"
epsg: "4326"

var_name: "equivalent_reflectivity_factor"
standard_name: "equivalent_reflectivity_factor"

mlcast_created_by: "Adrián García <agarciaa@aemet.es>"
mlcast_created_with: "https://github.com/mlcast-community/mlcast-dataset-ES-AEMET-reflectivity@v0.1.0"
mlcast_dataset_version: "0.1.0"
mlcast_dataset_identifier: "ES-AEMET-radar_reflectivity-ppi_ZAR"
mlcast_dataset_identifier_format: "{{country_code}}-{{entity}}-{{physical_variable}}-{{common_name}}"

license: "CC-BY-4.0"
institution: "Agencia Estatal de Meteorología (AEMET)"
source: "AEMET radar network"
attribution: "Data provided by AEMET"

compression_level: 5
time_chunk: 1
shard_time: 144
inspect: false
""".strip()

        with open(config_path, "w") as f:
            f.write(yaml_content)

        config_paths.append(str(config_path))

# Crear config_list.txt
list_path = CONFIG_DIR / "config_list.txt"
with open(list_path, "w") as f:
    for path in config_paths:
        f.write(path + "\n")

print(f"✔ Configs generados en: {CONFIG_DIR}")
print(f"✔ Lista: {list_path}")

# Crear zip
import zipfile

zip_path = CONFIG_DIR.parent / "monthly_v3_configs.zip"

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for path in config_paths:
        zipf.write(path, arcname=Path(path).name)
    zipf.write(list_path, arcname="config_list.txt")

print(f"✔ ZIP creado en: {zip_path}")