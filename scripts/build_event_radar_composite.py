from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from dask.diagnostics import ProgressBar
from zarr.codecs import ZstdCodec

from mlcast_dataset_es_aemet_reflectivity.pipeline_zarr2 import (
    RadarBuildConfig as RadarBuildConfigV2,
    run_pipeline as run_pipeline_v2,
)
from mlcast_dataset_es_aemet_reflectivity.pipeline_zarr3 import (
    RadarBuildConfigZarr3 as RadarBuildConfigV3,
    run_pipeline_zarr3 as run_pipeline_v3,
)


# =============================================================================
# CONFIG HELPERS
# =============================================================================

def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError("The YAML configuration must contain a dictionary.")

    return cfg


def get_radars(cfg: dict[str, Any]) -> list[str]:
    radars = cfg.get("radars")

    if radars is None:
        single_radar = cfg.get("radar")
        if single_radar is None:
            raise ValueError(
                "The config must contain either a non-empty 'radars' list "
                "or a single 'radar' value."
            )
        radars = [single_radar]

    if isinstance(radars, str):
        radars = [radars]

    radars = [str(radar) for radar in radars]

    if not radars:
        raise ValueError("The radar list is empty.")

    return radars


def get_input_var_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get("input_var_name", cfg.get("var_name", "reflectivity")))


def get_reflectivity_var_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get("reflectivity_var_name", "reflectivity_dbz"))


def get_rainfall_rate_var_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get("rainfall_rate_var_name", "rainfall_rate"))


def input_variable_is_dbz(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("input_variable_is_dbz", True))


# =============================================================================
# PATH HELPERS
# =============================================================================

def detect_zarr_format(path: Path) -> int:
    return 3 if (path / "zarr.json").exists() else 2


def remove_path(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def build_radar_zarr_path(cfg: dict[str, Any], radar: str) -> Path:
    start = cfg["fechaini"][:8]
    end = cfg["fechafin"][:8]
    root = Path(cfg["individual_zarr_root"])
    event_name = cfg["event_name"]
    product = str(cfg["imagen"]).lower()

    return root / radar / f"{event_name}_{product}_{radar}_{start}_{end}.zarr"


def build_radar_workdir(cfg: dict[str, Any], radar: str) -> Path:
    root = Path(cfg["workdir_root"])
    event_name = cfg["event_name"]
    return root / radar / event_name


def build_single_radar_cfg(cfg: dict[str, Any], radar: str) -> dict[str, Any]:
    radar_cfg = dict(cfg)

    input_var_name = get_input_var_name(cfg)

    radar_cfg["radar"] = radar
    radar_cfg["workdir"] = str(build_radar_workdir(cfg, radar))
    radar_cfg["zarr_out"] = str(build_radar_zarr_path(cfg, radar))

    # The ingestion pipeline writes the raw/input variable.
    radar_cfg["var_name"] = input_var_name
    radar_cfg["standard_name"] = cfg.get(
        "input_standard_name",
        cfg.get("standard_name", "equivalent_reflectivity_factor"),
    )

    radar_cfg["mlcast_dataset_identifier"] = (
        f"ES-AEMET-{input_var_name}-{str(cfg['imagen']).lower()}_{radar}"
    )

    return radar_cfg


# =============================================================================
# INDIVIDUAL RADAR BUILD
# =============================================================================

def build_config_v2(cfg: dict[str, Any]) -> RadarBuildConfigV2:
    return RadarBuildConfigV2(
        workdir=Path(cfg["workdir"]),
        zarr_out=Path(cfg["zarr_out"]),
        png_out=Path(cfg.get("png_out", "./radar_quicklook.png")),
        png_out_cartopy=Path(cfg.get("png_out_cartopy", "./radar_quicklook_cartopy.png")),
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
        compressor_name=cfg.get("compressor_name", "zstd"),
        compression_level=int(cfg["compression_level"]),
        blosc_shuffle=cfg.get("blosc_shuffle", "bitshuffle"),
        zarr_format=2,
        time_chunk=int(cfg["time_chunk"]),
        use_sharding=bool(cfg.get("use_sharding", False)),
        shard_time=cfg.get("shard_time", None),
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
        inspect=bool(cfg.get("inspect", False)),
    )


def build_individual_radar_zarr(cfg: dict[str, Any], radar: str) -> Path:
    radar_cfg = build_single_radar_cfg(cfg, radar)
    zarr_out = Path(radar_cfg["zarr_out"])

    if zarr_out.exists() and not bool(cfg.get("overwrite_individual", False)):
        print(f"[SKIP] Individual Zarr already exists for {radar}: {zarr_out}")
        return zarr_out

    if zarr_out.exists():
        print(f"[REMOVE] Existing individual Zarr for {radar}: {zarr_out}")
        remove_path(zarr_out)

    print(f"\n[BUILD] Radar {radar}")
    print(f"Output: {zarr_out}")

    zarr_version = int(cfg["zarr_version"])

    if zarr_version == 2:
        run_pipeline_v2(build_config_v2(radar_cfg))
    elif zarr_version == 3:
        run_pipeline_v3(build_config_v3(radar_cfg))
    else:
        raise ValueError(f"Unsupported zarr_version: {zarr_version}")

    return zarr_out


# =============================================================================
# RADAR OPENING AND CONVERSION
# =============================================================================

def open_radar_zarr(path: Path, var_name: str) -> xr.Dataset:
    zarr_format = detect_zarr_format(path)

    ds = xr.open_zarr(
        path,
        consolidated=False if zarr_format == 3 else True,
        zarr_format=zarr_format,
        chunks={"time": 1},
    )

    if var_name not in ds:
        raise KeyError(
            f"Variable {var_name!r} not found in {path}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    return ds


def select_event_period(ds: xr.Dataset, cfg: dict[str, Any]) -> xr.Dataset:
    start = pd.to_datetime(cfg["fechaini"])
    end = pd.to_datetime(cfg["fechafin"])
    return ds.sel(time=slice(start, end))


def dbz_to_rainfall_rate_mm_h(
    dbz: xr.DataArray,
    marshall_palmer_a: float,
    marshall_palmer_b: float,
) -> xr.DataArray:
    z_linear = 10.0 ** (dbz / 10.0)
    rainfall_rate = (z_linear / marshall_palmer_a) ** (1.0 / marshall_palmer_b)

    rainfall_rate = rainfall_rate.astype("float32")
    rainfall_rate.attrs = {
        "long_name": (
            "Rainfall rate derived from radar reflectivity using the "
            "Marshall-Palmer Z-R relation"
        ),
        "standard_name": "rainfall_rate",
        "units": "mm h-1",
        "source_units": "dBZ",
        "conversion_formula": "Z = a * R**b; R = (10**(dBZ / 10) / a)**(1 / b)",
        "marshall_palmer_a": float(marshall_palmer_a),
        "marshall_palmer_b": float(marshall_palmer_b),
    }

    return rainfall_rate


def prepare_radar_fields(
    ds: xr.Dataset,
    cfg: dict[str, Any],
) -> tuple[xr.DataArray, xr.DataArray]:
    input_name = get_input_var_name(cfg)
    reflectivity_name = get_reflectivity_var_name(cfg)
    rainfall_name = get_rainfall_rate_var_name(cfg)

    source = ds[input_name]

    if "time" not in source.dims:
        raise ValueError(f"Variable {input_name!r} has no time dimension.")

    if "x" not in source.coords or "y" not in source.coords:
        raise KeyError("x/y coordinates not found in the source variable.")

    if input_variable_is_dbz(cfg):
        reflectivity_dbz = source.astype("float32")
        reflectivity_dbz.name = reflectivity_name
        reflectivity_dbz.attrs = {
            "long_name": "Radar reflectivity factor in logarithmic units",
            "standard_name": "equivalent_reflectivity_factor",
            "units": "dBZ",
            "coordinates": "lat lon",
            "grid_mapping": "spatial_ref",
        }

        rainfall_rate = dbz_to_rainfall_rate_mm_h(
            reflectivity_dbz,
            marshall_palmer_a=float(cfg.get("marshall_palmer_a", 200.0)),
            marshall_palmer_b=float(cfg.get("marshall_palmer_b", 1.6)),
        )
        rainfall_rate.name = rainfall_name

    else:
        rainfall_rate = source.astype("float32")
        rainfall_rate.name = rainfall_name
        rainfall_rate.attrs = {
            "long_name": cfg.get("long_name", "Radar rainfall rate"),
            "standard_name": "rainfall_rate",
            "units": cfg.get("units", "mm h-1"),
            "coordinates": "lat lon",
            "grid_mapping": "spatial_ref",
        }

        reflectivity_dbz = xr.full_like(rainfall_rate, np.nan).astype("float32")
        reflectivity_dbz.name = reflectivity_name
        reflectivity_dbz.attrs = {
            "long_name": "Radar reflectivity factor not available",
            "standard_name": "equivalent_reflectivity_factor",
            "units": "dBZ",
            "description": (
                "Input variable was configured as rainfall rate, so no original "
                "reflectivity field was available."
            ),
            "coordinates": "lat lon",
            "grid_mapping": "spatial_ref",
        }

    return reflectivity_dbz, rainfall_rate


# =============================================================================
# COMPOSITE
# =============================================================================

def get_reference_grid(
    zarr_paths: dict[str, Path],
    cfg: dict[str, Any],
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray | None, xr.DataArray | None]:
    reference_radar = cfg["composite"]["reference_radar"]
    input_var_name = get_input_var_name(cfg)

    if reference_radar not in zarr_paths:
        raise ValueError(
            f"reference_radar={reference_radar!r} is not among selected radars: "
            f"{list(zarr_paths)}"
        )

    ds_ref = open_radar_zarr(zarr_paths[reference_radar], var_name=input_var_name)

    try:
        ds_ref = select_event_period(ds_ref, cfg)

        if "x" not in ds_ref.coords or "y" not in ds_ref.coords:
            raise KeyError("Reference radar dataset must contain x and y coordinates.")

        x_ref = ds_ref["x"].copy()
        y_ref = ds_ref["y"].copy()

        lat_ref = ds_ref["lat"].copy() if "lat" in ds_ref else None
        lon_ref = ds_ref["lon"].copy() if "lon" in ds_ref else None

    finally:
        ds_ref.close()

    return x_ref, y_ref, lat_ref, lon_ref


def regrid_to_reference(
    da: xr.DataArray,
    x_ref: xr.DataArray,
    y_ref: xr.DataArray,
) -> xr.DataArray:
    return da.interp(
        x=x_ref,
        y=y_ref,
        method="nearest",
        kwargs={"fill_value": np.nan},
    )


def composite_along_radar(stacked: xr.DataArray, method: str) -> xr.DataArray:
    if method == "max":
        return stacked.max("radar", skipna=True)
    if method == "mean":
        return stacked.mean("radar", skipna=True)
    if method == "median":
        return stacked.median("radar", skipna=True)

    raise ValueError(
        f"Unsupported composite method: {method!r}. "
        "Allowed values are: max, mean, median."
    )


def create_composite_dataset(
    zarr_paths: dict[str, Path],
    cfg: dict[str, Any],
) -> xr.Dataset:
    input_var_name = get_input_var_name(cfg)
    reflectivity_name = get_reflectivity_var_name(cfg)
    rainfall_name = get_rainfall_rate_var_name(cfg)
    method = cfg["composite"].get("method", "max").lower()

    x_ref, y_ref, lat_ref, lon_ref = get_reference_grid(zarr_paths, cfg)

    reflectivity_arrays = []
    rainfall_arrays = []
    radar_names = []

    for radar, path in zarr_paths.items():
        print(f"\n[OPEN] {radar}: {path}")
        ds = open_radar_zarr(path, var_name=input_var_name)

        try:
            ds = select_event_period(ds, cfg)

            reflectivity_dbz, rainfall_rate = prepare_radar_fields(ds, cfg)

            reflectivity_on_ref = regrid_to_reference(
                reflectivity_dbz,
                x_ref=x_ref,
                y_ref=y_ref,
            ).expand_dims(radar=[radar])

            rainfall_on_ref = regrid_to_reference(
                rainfall_rate,
                x_ref=x_ref,
                y_ref=y_ref,
            ).expand_dims(radar=[radar])

            reflectivity_arrays.append(reflectivity_on_ref)
            rainfall_arrays.append(rainfall_on_ref)
            radar_names.append(radar)

        finally:
            ds.close()

    if not rainfall_arrays:
        raise ValueError("No radar arrays were created.")

    stacked_reflectivity = xr.concat(reflectivity_arrays, dim="radar")
    stacked_rainfall = xr.concat(rainfall_arrays, dim="radar")

    stacked_reflectivity = stacked_reflectivity.assign_coords(radar=("radar", radar_names))
    stacked_rainfall = stacked_rainfall.assign_coords(radar=("radar", radar_names))

    rainfall_composite = composite_along_radar(stacked_rainfall, method=method)
    reflectivity_composite = composite_along_radar(stacked_reflectivity, method=method)

    radar_count = stacked_rainfall.notnull().sum("radar").astype("int16")

    dominant_radar_index = stacked_rainfall.argmax("radar", skipna=True).astype("int16")
    dominant_radar_index = dominant_radar_index.where(radar_count > 0, -1)

    rainfall_composite = rainfall_composite.astype("float32")
    rainfall_composite.name = rainfall_name
    rainfall_composite.attrs = {
        "long_name": "Composite radar-estimated rainfall rate",
        "standard_name": "rainfall_rate",
        "units": "mm h-1",
        "coordinates": "lat lon",
        "grid_mapping": "spatial_ref",
        "composite_method": method,
        "selected_radars": ",".join(radar_names),
        "source_variable": input_var_name,
        "input_variable_is_dbz": str(input_variable_is_dbz(cfg)),
    }

    if input_variable_is_dbz(cfg):
        rainfall_composite.attrs.update(
            {
                "conversion_formula": (
                    "Z = a * R**b; "
                    "R = (10**(dBZ / 10) / a)**(1 / b)"
                ),
                "marshall_palmer_a": float(cfg.get("marshall_palmer_a", 200.0)),
                "marshall_palmer_b": float(cfg.get("marshall_palmer_b", 1.6)),
            }
        )

    reflectivity_composite = reflectivity_composite.astype("float32")
    reflectivity_composite.name = reflectivity_name
    reflectivity_composite.attrs = {
        "long_name": "Composite radar reflectivity factor",
        "standard_name": "equivalent_reflectivity_factor",
        "units": "dBZ",
        "coordinates": "lat lon",
        "grid_mapping": "spatial_ref",
        "composite_method": method,
        "selected_radars": ",".join(radar_names),
        "source_variable": input_var_name,
    }

    ds_out = xr.Dataset(
        data_vars={
            rainfall_name: rainfall_composite,
            reflectivity_name: reflectivity_composite,
            "radar_count_available": radar_count,
            "dominant_radar_index": dominant_radar_index,
        },
        coords={
            "time": rainfall_composite["time"],
            "y": y_ref,
            "x": x_ref,
        },
        attrs={
            "title": f"Radar composite for {cfg['event_name']}",
            "summary": (
                "Multi-radar event composite. If the input variable is configured "
                "as dBZ, rainfall_rate is derived using the Marshall-Palmer Z-R "
                "relation before compositing."
            ),
            "event_name": cfg["event_name"],
            "source": cfg["source"],
            "institution": cfg["institution"],
            "license": cfg["license"],
            "attribution": cfg["attribution"],
            "selected_radars": ",".join(radar_names),
            "composite_method": method,
            "input_var_name": input_var_name,
            "input_variable_is_dbz": str(input_variable_is_dbz(cfg)),
            "rainfall_rate_var_name": rainfall_name,
            "reflectivity_var_name": reflectivity_name,
            "mlcast_created_by": cfg["mlcast_created_by"],
            "mlcast_created_with": cfg["mlcast_created_with"],
            "mlcast_dataset_version": cfg["mlcast_dataset_version"],
            "mlcast_dataset_identifier": cfg["mlcast_dataset_identifier"],
            "mlcast_dataset_identifier_format": cfg["mlcast_dataset_identifier_format"],
        },
    )

    if "spatial_ref" in ds:
        # Copying from the last opened dataset is acceptable here because all
        # radars are expected to have been generated in the same CRS.
        ds_out["spatial_ref"] = ds["spatial_ref"]

    if lat_ref is not None and lon_ref is not None:
        ds_out = ds_out.assign_coords(
            lat=(lat_ref.dims, lat_ref.data.astype(np.float32)),
            lon=(lon_ref.dims, lon_ref.data.astype(np.float32)),
        )

        ds_out["lat"].attrs = {
            "standard_name": "latitude",
            "long_name": "latitude",
            "units": "degrees_north",
        }

        ds_out["lon"].attrs = {
            "standard_name": "longitude",
            "long_name": "longitude",
            "units": "degrees_east",
        }

    ds_out["radar_count_available"].attrs = {
        "long_name": "Number of radars with valid data at each grid point",
        "units": "1",
    }

    ds_out["dominant_radar_index"].attrs = {
        "long_name": "Index of radar contributing the selected rainfall-rate composite value",
        "description": (
            "Index along the radar list used internally to identify which radar "
            "provided the selected composite value. -1 means no radar data available."
        ),
        "radar_order": ",".join(radar_names),
        "units": "1",
    }

    return ds_out


# =============================================================================
# OUTPUT
# =============================================================================

def build_encoding(ds: xr.Dataset, cfg: dict[str, Any]) -> dict[str, Any]:
    compressor = ZstdCodec(level=int(cfg["compression_level"]))

    time_chunk = int(cfg["time_chunk"])
    shard_time = int(cfg["shard_time"])

    y_size = ds.sizes["y"]
    x_size = ds.sizes["x"]

    encoding: dict[str, Any] = {}

    for name, da in ds.data_vars.items():
        if da.dims == ("time", "y", "x"):
            dtype = np.float32 if np.issubdtype(da.dtype, np.floating) else da.dtype

            encoding[name] = {
                "compressors": (compressor,),
                "chunks": (time_chunk, y_size, x_size),
                "shards": (shard_time, y_size, x_size),
                "dtype": dtype,
            }

            if np.issubdtype(da.dtype, np.floating):
                encoding[name]["_FillValue"] = np.float32(np.nan)

    if "lat" in ds.coords:
        encoding["lat"] = {
            "compressors": (compressor,),
            "chunks": (y_size, x_size),
            "dtype": np.float32,
            "_FillValue": np.float32(np.nan),
        }

    if "lon" in ds.coords:
        encoding["lon"] = {
            "compressors": (compressor,),
            "chunks": (y_size, x_size),
            "dtype": np.float32,
            "_FillValue": np.float32(np.nan),
        }

    return encoding


def write_composite_zarr(ds: xr.Dataset, cfg: dict[str, Any]) -> None:
    output_zarr = Path(cfg["composite_zarr_out"])

    if output_zarr.exists():
        if bool(cfg.get("overwrite_composite", False)):
            print(f"[REMOVE] Existing composite Zarr: {output_zarr}")
            remove_path(output_zarr)
        else:
            raise FileExistsError(
                f"Composite Zarr already exists: {output_zarr}. "
                "Set overwrite_composite: true to replace it."
            )

    output_zarr.parent.mkdir(parents=True, exist_ok=True)

    encoding = build_encoding(ds, cfg)

    print(f"\n[WRITE] Composite Zarr: {output_zarr}")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Encoding variables: {list(encoding)}")

    with ProgressBar():
        ds.to_zarr(
            output_zarr,
            mode="w",
            consolidated=False,
            zarr_format=int(cfg["zarr_version"]),
            encoding=encoding,
            compute=True,
            align_chunks=True,
            safe_chunks=False,
        )

    print(f"\n[OK] Composite written to: {output_zarr}")


# =============================================================================
# PIPELINE
# =============================================================================

def run_event_pipeline(cfg: dict[str, Any]) -> None:
    radars = get_radars(cfg)

    zarr_paths: dict[str, Path] = {}

    for radar in radars:
        zarr_path = build_individual_radar_zarr(cfg, radar)
        zarr_paths[radar] = zarr_path

    ds_composite = create_composite_dataset(zarr_paths, cfg)
    write_composite_zarr(ds_composite, cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build individual radar Zarr stores for a short event and create "
            "a multi-radar composite on a common grid. The script can convert "
            "input dBZ reflectivity to rainfall rate before compositing."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the event YAML configuration.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    run_event_pipeline(cfg)


if __name__ == "__main__":
    main()
