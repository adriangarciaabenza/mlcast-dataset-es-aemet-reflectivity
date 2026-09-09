from __future__ import annotations

import argparse
import zipfile
from datetime import datetime
from pathlib import Path


# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_PROJECT_DIR = Path("/perm/pred/std/ML/MLCAST/mlcast-dataset-ES-AEMET-reflectivity")
DEFAULT_CONFIG_DIR = DEFAULT_PROJECT_DIR / "configs/monthly_SRI_v1"
DEFAULT_OUTPUT_DIR = Path("/lustre/utmp/std/MLCAST_radar_data/outputs/monthly_SRI_v1")
DEFAULT_START_YEAR = 2020
DEFAULT_END_YEAR = 2024
DEFAULT_CREATE_ZIP = True


# =============================================================================
# HELPERS
# =============================================================================


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_monthly_yaml_content(
    project_dir: Path,
    output_dir: Path,
    year: int,
    month: int,
) -> tuple[str, str]:
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 1, 1)

    name = f"{year}_{month:02d}"

    yaml_content = f"""
zarr_version: 3

workdir: "{project_dir}/workdir/monthly_SRI_v1/{name}"
zarr_out: "{output_dir}/radar_{name}.zarr"

fechaini: "{start.strftime('%Y%m%dT%H%M%S')}"
fechafin: "{end.strftime('%Y%m%dT%H%M%S')}"

imagen: "SRI"
configuracion: "Z_240"
radar: "ZAR"
epsg: "25830"

var_name: "rainfall_rate"
standard_name: "rainfall_flux"

mlcast_created_by: "Jaime Castro <jcastroa@aemet.es>; Adrián García <agarciaa@aemet.es>"
mlcast_created_with: "https://github.com/mlcast-community/mlcast-dataset-ES-AEMET-reflectivity@v0.1.0"
mlcast_dataset_version: "0.1.0"
mlcast_dataset_identifier: "ES-AEMET-rainfall_rate-sri_ZAR"
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

    return name, yaml_content


def generate_monthly_configs(
    project_dir: Path,
    config_dir: Path,
    output_dir: Path,
    start_year: int,
    end_year: int,
) -> tuple[list[Path], Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_paths: list[Path] = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            name, yaml_content = build_monthly_yaml_content(
                project_dir=project_dir,
                output_dir=output_dir,
                year=year,
                month=month,
            )

            config_path = config_dir / f"{name}.yaml"
            with config_path.open("w", encoding="utf-8") as file:
                file.write(yaml_content)

            config_paths.append(config_path)

    list_path = config_dir / "config_list.txt"
    with list_path.open("w", encoding="utf-8") as file:
        for path in config_paths:
            file.write(str(path) + "\n")

    return config_paths, list_path


def create_zip_file(
    config_paths: list[Path],
    list_path: Path,
    zip_path: Path,
) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path in config_paths:
            zip_file.write(path, arcname=path.name)
        zip_file.write(list_path, arcname="config_list.txt")


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate monthly YAML configuration files for the radar Zarr pipeline"
    )

    parser.add_argument("--project_dir", type=str, default=None)
    parser.add_argument("--config_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--start_year", type=int, default=None)
    parser.add_argument("--end_year", type=int, default=None)

    parser.add_argument("--create_zip", type=str2bool, default=None)

    return parser


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_dir = Path(args.project_dir) if args.project_dir else DEFAULT_PROJECT_DIR
    config_dir = Path(args.config_dir) if args.config_dir else DEFAULT_CONFIG_DIR
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    start_year = args.start_year if args.start_year is not None else DEFAULT_START_YEAR
    end_year = args.end_year if args.end_year is not None else DEFAULT_END_YEAR
    create_zip = args.create_zip if args.create_zip is not None else DEFAULT_CREATE_ZIP

    if end_year < start_year:
        raise ValueError(
            f"end_year must be greater than or equal to start_year, got {start_year} and {end_year}"
        )

    config_paths, list_path = generate_monthly_configs(
        project_dir=project_dir,
        config_dir=config_dir,
        output_dir=output_dir,
        start_year=start_year,
        end_year=end_year,
    )

    print(f"Configs generated in: {config_dir}")
    print(f"Config list written to: {list_path}")
    print(f"Number of config files generated: {len(config_paths)}")

    if create_zip:
        zip_path = config_dir.parent / "monthly_v3_configs.zip"
        create_zip_file(
            config_paths=config_paths,
            list_path=list_path,
            zip_path=zip_path,
        )
        print(f"ZIP file created at: {zip_path}")


if __name__ == "__main__":
    main()