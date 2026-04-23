from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from mlcast_dataset_es_aemet_reflectivity.pipeline_zarr2 import (
    RadarBuildConfig as RadarBuildConfigV2,
    run_pipeline as run_pipeline_v2,
)
from mlcast_dataset_es_aemet_reflectivity.pipeline_zarr3 import (
    RadarBuildConfigZarr3 as RadarBuildConfigV3,
    run_pipeline_zarr3 as run_pipeline_v3,
)


# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULTS: dict[str, Any] = {
    # -------------------------------------------------------------------------
    # Zarr selector
    # -------------------------------------------------------------------------
    "zarr_version": 3,

    # -------------------------------------------------------------------------
    # Common paths
    # -------------------------------------------------------------------------
    "workdir": "./workdir_zarr",
    "zarr_out": "./ES-AEMET-radar_reflectivity-ppi_ZAR.zarr",

    # -------------------------------------------------------------------------
    # Only used by v2
    # -------------------------------------------------------------------------
    "png_out": "./radar_quicklook.png",
    "png_out_cartopy": "./radar_quicklook_cartopy.png",

    # -------------------------------------------------------------------------
    # Time range
    # -------------------------------------------------------------------------
    "fechaini": "20241001T000000",
    "fechafin": "20241002T000000",

    # -------------------------------------------------------------------------
    # Radar metadata
    # -------------------------------------------------------------------------
    "imagen": "PPI",
    "configuracion": "Z_005_240",
    "radar": "ZAR",
    "epsg": "4326",
    "var_name": "equivalent_reflectivity_factor",
    "standard_name": "equivalent_reflectivity_factor",

    # -------------------------------------------------------------------------
    # MLCAST metadata
    # -------------------------------------------------------------------------
    "mlcast_created_by": "Adrián García <agarciaa@aemet.es>",
    "mlcast_created_with": (
        "https://github.com/mlcast-community/"
        "mlcast-dataset-ES-AEMET-reflectivity@v0.1.0"
    ),
    "mlcast_dataset_version": "0.1.0",
    "mlcast_dataset_identifier": "ES-AEMET-radar_reflectivity-ppi_ZAR",
    "mlcast_dataset_identifier_format": (
        "{country_code}-{entity}-{physical_variable}-{common_name}"
    ),

    # -------------------------------------------------------------------------
    # LICENSE / ATTRIBUTION
    # -------------------------------------------------------------------------
    "license": "CC-BY-4.0",
    "institution": "Agencia Estatal de Meteorología (AEMET)",
    "source": "AEMET radar network",
    "attribution": "Data provided by AEMET",

    # -------------------------------------------------------------------------
    # Compression / chunking
    # -------------------------------------------------------------------------
    "compression_level": 5,
    "time_chunk": 1,

    # -------------------------------------------------------------------------
    # v2-only compression fields
    # -------------------------------------------------------------------------
    "compressor_name": "zstd",
    "blosc_shuffle": "bitshuffle",
    "zarr_format": 2,
    "use_sharding": False,

    # -------------------------------------------------------------------------
    # v3-only / optional
    # -------------------------------------------------------------------------
    "shard_time": 144,
    "inspect": False,
}


# =============================================================================
# HELPERS
# =============================================================================

def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    value = value.strip().lower()
    if value in {"true", "1", "yes", "y", "si", "sí"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a key-value dictionary")

    return data


def cli_overrides_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    """
    Return only CLI arguments that were explicitly provided.

    Final precedence: CLI > YAML > DEFAULTS
    """
    result: dict[str, Any] = {}

    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            result[key] = value

    return result


def build_final_config_dict(
    defaults: dict[str, Any],
    yaml_config: dict[str, Any],
    cli_config: dict[str, Any],
) -> dict[str, Any]:
    final = defaults.copy()
    final.update(yaml_config)
    final.update(cli_config)
    return final


def validate_config(cfg: dict[str, Any]) -> None:
    zarr_version = int(cfg["zarr_version"])
    if zarr_version not in {2, 3}:
        raise ValueError(
            f"zarr_version must be 2 or 3, got: {zarr_version}"
        )


def build_config_v2(cfg: dict[str, Any]) -> RadarBuildConfigV2:
    shard_time = cfg.get("shard_time")
    if shard_time in ("", "none", "None"):
        shard_time = None

    return RadarBuildConfigV2(
        workdir=Path(cfg["workdir"]),
        zarr_out=Path(cfg["zarr_out"]),
        png_out=Path(cfg["png_out"]),
        png_out_cartopy=Path(cfg["png_out_cartopy"]),
        fechaini=cfg["fechaini"],
        fechafin=cfg["fechafin"],
        imagen=cfg["imagen"],
        configuracion=cfg["configuracion"],
        radar=cfg["radar"],
        epsg=str(cfg["epsg"]),
        var_name=cfg["var_name"],
        standard_name=cfg["standard_name"],
        mlcast_created_by=cfg["mlcast_created_by"],
        mlcast_created_with=cfg["mlcast_created_with"],
        mlcast_dataset_version=cfg["mlcast_dataset_version"],
        mlcast_dataset_identifier=cfg["mlcast_dataset_identifier"],
        mlcast_dataset_identifier_format=cfg["mlcast_dataset_identifier_format"],
        license=cfg["license"],
        institution=cfg["institution"],
        source=cfg["source"],
        attribution=cfg["attribution"],
        compressor_name=cfg["compressor_name"],
        compression_level=int(cfg["compression_level"]),
        blosc_shuffle=cfg["blosc_shuffle"],
        zarr_format=int(cfg["zarr_format"]),
        time_chunk=int(cfg["time_chunk"]),
        use_sharding=bool(cfg["use_sharding"]),
        shard_time=None if shard_time is None else int(shard_time),
    )


def build_config_v3(cfg: dict[str, Any]) -> RadarBuildConfigV3:
    return RadarBuildConfigV3(
        workdir=Path(cfg["workdir"]),
        zarr_out=Path(cfg["zarr_out"]),
        fechaini=cfg["fechaini"],
        fechafin=cfg["fechafin"],
        imagen=cfg["imagen"],
        configuracion=cfg["configuracion"],
        radar=cfg["radar"],
        epsg=str(cfg["epsg"]),
        var_name=cfg["var_name"],
        standard_name=cfg["standard_name"],
        mlcast_created_by=cfg["mlcast_created_by"],
        mlcast_created_with=cfg["mlcast_created_with"],
        mlcast_dataset_version=cfg["mlcast_dataset_version"],
        mlcast_dataset_identifier=cfg["mlcast_dataset_identifier"],
        mlcast_dataset_identifier_format=cfg["mlcast_dataset_identifier_format"],
        license=cfg["license"],
        institution=cfg["institution"],
        source=cfg["source"],
        attribution=cfg["attribution"],
        compression_level=int(cfg["compression_level"]),
        time_chunk=int(cfg["time_chunk"]),
        shard_time=int(cfg["shard_time"]),
        inspect=bool(cfg["inspect"]),
    )


def run_from_config(cfg: dict[str, Any]) -> None:
    zarr_version = int(cfg["zarr_version"])

    if zarr_version == 2:
        config = build_config_v2(cfg)
        run_pipeline_v2(config)
    elif zarr_version == 3:
        config = build_config_v3(cfg)
        run_pipeline_v3(config)
    else:
        raise ValueError(f"Unsupported Zarr version: {zarr_version}")


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Single entry point to build radar datasets in Zarr v2 or v3 "
            "using defaults, YAML, and CLI overrides."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the YAML configuration file.",
    )

    parser.add_argument("--zarr_version", type=int, default=None)

    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--zarr_out", type=str, default=None)
    parser.add_argument("--png_out", type=str, default=None)
    parser.add_argument("--png_out_cartopy", type=str, default=None)

    parser.add_argument("--fechaini", type=str, default=None)
    parser.add_argument("--fechafin", type=str, default=None)

    parser.add_argument("--imagen", type=str, default=None)
    parser.add_argument("--configuracion", type=str, default=None)
    parser.add_argument("--radar", type=str, default=None)
    parser.add_argument("--epsg", type=str, default=None)

    parser.add_argument("--var_name", type=str, default=None)
    parser.add_argument("--standard_name", type=str, default=None)

    parser.add_argument("--mlcast_created_by", type=str, default=None)
    parser.add_argument("--mlcast_created_with", type=str, default=None)
    parser.add_argument("--mlcast_dataset_version", type=str, default=None)
    parser.add_argument("--mlcast_dataset_identifier", type=str, default=None)
    parser.add_argument("--mlcast_dataset_identifier_format", type=str, default=None)

    parser.add_argument("--license", type=str, default=None)
    parser.add_argument("--institution", type=str, default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--attribution", type=str, default=None)

    parser.add_argument("--compression_level", type=int, default=None)
    parser.add_argument("--time_chunk", type=int, default=None)

    # v2
    parser.add_argument("--compressor_name", type=str, default=None)
    parser.add_argument("--blosc_shuffle", type=str, default=None)
    parser.add_argument("--zarr_format", type=int, default=None)
    parser.add_argument("--use_sharding", type=str2bool, default=None)

    # Common / optional depending on version
    parser.add_argument("--shard_time", type=int, default=None)

    # v3
    parser.add_argument("--inspect", type=str2bool, default=None)

    return parser


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    yaml_config = load_yaml_config(args.config)
    cli_config = cli_overrides_to_dict(args)

    final_cfg = build_final_config_dict(
        defaults=DEFAULTS,
        yaml_config=yaml_config,
        cli_config=cli_config,
    )

    validate_config(final_cfg)
    run_from_config(final_cfg)