from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from zarr.codecs import ZstdCodec


# =============================================================================
# DEFAULT CONFIG
# =============================================================================

DEFAULT_INPUT_DIR = Path("/lustre/utmp/std/MLCAST_radar_data/outputs/monthly_v4")
DEFAULT_OUTPUT_ZARR = Path(
    "/lustre/utmp/std/MLCAST_radar_data/"
    "ES-AEMET-radar_reflectivity-ppi_ZAR_2020-2024_v4.zarr"
)

DEFAULT_VAR_NAME = "reflectivity"
DEFAULT_EPSG = "EPSG:25830"

TIME_FREQ = "10min"
TIME_CHUNK = 1
SHARD_TIME = 144
COMPRESSION_LEVEL = 12
ZARR_FORMAT = 3

VAR_NAME = DEFAULT_VAR_NAME
EPSG = DEFAULT_EPSG


# =============================================================================
# HELPERS
# =============================================================================

def normalize_epsg(epsg: str) -> str:
    epsg = str(epsg).strip().upper()
    if epsg.startswith("EPSG:"):
        return epsg
    return f"EPSG:{epsg}"


def is_projected_epsg(epsg: str) -> bool:
    return normalize_epsg(epsg) != "EPSG:4326"


def _remove_path(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _find_input_zarrs(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_dir}")

    zarrs = sorted([p for p in input_dir.glob("*.zarr") if p.is_dir()])

    if not zarrs:
        raise ValueError(f"No .zarr directories were found in: {input_dir}")

    return zarrs


def _open_monthly_zarr(path: Path) -> xr.Dataset:
    ds = xr.open_zarr(path, consolidated=False)

    if VAR_NAME not in ds.data_vars:
        raise ValueError(f"{path}: does not contain variable '{VAR_NAME}'")

    return ds


def _ensure_lat_lon_as_coords(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()

    lat_data = None
    lon_data = None

    if "lat" in ds:
        lat_data = ds["lat"]
    elif "latitude" in ds:
        lat_data = ds["latitude"]

    if "lon" in ds:
        lon_data = ds["lon"]
    elif "longitude" in ds:
        lon_data = ds["longitude"]

    if lat_data is None or lon_data is None:
        raise ValueError("lat/lon coordinates were not found in the dataset.")

    if "time" in lat_data.dims:
        lat_data = lat_data.isel(time=0, drop=True)
    if "time" in lon_data.dims:
        lon_data = lon_data.isel(time=0, drop=True)

    to_drop = [
        v for v in ["lat", "lon", "latitude", "longitude"]
        if v in ds.data_vars
    ]
    if to_drop:
        ds = ds.drop_vars(to_drop)

    ds = ds.assign_coords(
        lat=(lat_data.dims, lat_data.data.astype(np.float32)),
        lon=(lon_data.dims, lon_data.data.astype(np.float32)),
    )

    ds["lat"].attrs = {
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
    }
    ds["lon"].attrs = {
        "standard_name": "longitude",
        "long_name": "longitude",
        "units": "degrees_east",
    }

    return ds


def _sort_and_drop_duplicate_times(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.sortby("time")
    time_index = pd.DatetimeIndex(ds["time"].values)
    _, unique_idx = np.unique(time_index.values, return_index=True)
    unique_idx = np.sort(unique_idx)
    return ds.isel(time=unique_idx)


def _set_cf_attrs(ds: xr.Dataset, epsg: str) -> xr.Dataset:
    ds = ds.copy()

    epsg_norm = normalize_epsg(epsg)
    projected = is_projected_epsg(epsg_norm)

    ds[VAR_NAME] = ds[VAR_NAME].astype(np.float32)
    ds[VAR_NAME].attrs.update(
        {
            "var_name": VAR_NAME,
            "long_name": "Radar reflectivity",
            "standard_name": "equivalent_reflectivity_factor",
            "units": "dBZ",
            "grid_mapping": "spatial_ref",
            "coordinates": "lat lon",
        }
    )

    if "time" in ds.coords:
        ds["time"].attrs.update(
            {
                "standard_name": "time",
                "long_name": "time",
            }
        )

    if "x" in ds.coords and "y" in ds.coords:
        if projected:
            ds["x"].attrs.update(
                {
                    "standard_name": "projection_x_coordinate",
                    "long_name": "x coordinate of projection",
                    "units": "m",
                    "axis": "X",
                }
            )
            ds["y"].attrs.update(
                {
                    "standard_name": "projection_y_coordinate",
                    "long_name": "y coordinate of projection",
                    "units": "m",
                    "axis": "Y",
                }
            )
        else:
            ds["x"].attrs.update(
                {
                    "standard_name": "longitude",
                    "long_name": "longitude",
                    "units": "degrees_east",
                    "axis": "X",
                }
            )
            ds["y"].attrs.update(
                {
                    "standard_name": "latitude",
                    "long_name": "latitude",
                    "units": "degrees_north",
                    "axis": "Y",
                }
            )

    if "spatial_ref" in ds:
        ds["spatial_ref"].attrs.setdefault("epsg_code", epsg_norm)
        ds["spatial_ref"].attrs.setdefault(
            "grid_mapping_name",
            "transverse_mercator" if projected else "latitude_longitude",
        )

    return ds


def _clear_inherited_encodings(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    for name in ds.variables:
        ds[name].encoding = {}
    return ds


def _validate_chunk_shard_settings() -> None:
    if TIME_CHUNK != 1:
        raise ValueError("To pass the validator, TIME_CHUNK must be 1.")
    if SHARD_TIME < TIME_CHUNK:
        raise ValueError("SHARD_TIME must be greater than or equal to TIME_CHUNK.")
    if SHARD_TIME % TIME_CHUNK != 0:
        raise ValueError("SHARD_TIME must be a multiple of TIME_CHUNK.")


def _report_layout(n_time: int) -> None:
    n_chunks = int(np.ceil(n_time / TIME_CHUNK))
    n_shards = int(np.ceil(n_time / SHARD_TIME))

    print("\n=== LAYOUT ===")
    print(f"time size      : {n_time}")
    print(f"time chunk     : {TIME_CHUNK}")
    print(f"time shard     : {SHARD_TIME}")
    print(f"logical chunks : ~{n_chunks}")
    print(f"physical shards: ~{n_shards}")


def _build_encoding(y_size: int, x_size: int) -> dict:
    compressor = ZstdCodec(level=COMPRESSION_LEVEL)

    _validate_chunk_shard_settings()

    return {
        VAR_NAME: {
            "compressors": (compressor,),
            "chunks": (TIME_CHUNK, y_size, x_size),
            "shards": (SHARD_TIME, y_size, x_size),
            "dtype": np.float32,
            "_FillValue": np.float32(np.nan),
        },
        "lat": {
            "compressors": (compressor,),
            "chunks": (y_size, x_size),
            "dtype": np.float32,
            "_FillValue": np.float32(np.nan),
        },
        "lon": {
            "compressors": (compressor,),
            "chunks": (y_size, x_size),
            "dtype": np.float32,
            "_FillValue": np.float32(np.nan),
        },
        "x": {
            "compressors": (compressor,),
            "chunks": (x_size,),
            "dtype": np.float64,
        },
        "y": {
            "compressors": (compressor,),
            "chunks": (y_size,),
            "dtype": np.float64,
        },
    }


# =============================================================================
# STEP 1. SCAN GLOBAL TIMES
# =============================================================================

def _collect_global_time_info(
    zarrs: list[Path],
) -> tuple[pd.DatetimeIndex, set[np.datetime64]]:
    all_times = []
    union_times: set[np.datetime64] = set()

    for i, path in enumerate(zarrs, start=1):
        print(f"[scan {i}/{len(zarrs)}] {path}")
        ds = _open_monthly_zarr(path)
        try:
            ds = _sort_and_drop_duplicate_times(ds)
            t = pd.DatetimeIndex(ds["time"].values)

            if len(t) == 0:
                continue

            all_times.append((t[0], t[-1]))
            union_times.update(t.values)
        finally:
            ds.close()

    if not all_times:
        raise ValueError("No valid times were found in the input Zarr stores.")

    global_start = min(t0 for t0, _ in all_times)
    global_end = max(t1 for _, t1 in all_times)

    full_time = pd.date_range(start=global_start, end=global_end, freq=TIME_FREQ)
    return full_time, union_times


# =============================================================================
# STEP 2. CREATE GLOBAL TEMPLATE
# =============================================================================

def _build_template_dataset(
    first_ds: xr.Dataset,
    full_time: pd.DatetimeIndex,
    epsg: str,
) -> xr.Dataset:
    epsg_norm = normalize_epsg(epsg)
    projected = is_projected_epsg(epsg_norm)

    first_ds = _ensure_lat_lon_as_coords(first_ds)
    first_ds = _sort_and_drop_duplicate_times(first_ds)
    first_ds = _set_cf_attrs(first_ds, epsg=epsg_norm)
    first_ds = _clear_inherited_encodings(first_ds)

    y_size = first_ds.sizes["y"]
    x_size = first_ds.sizes["x"]

    data = da.empty(
        (len(full_time), y_size, x_size),
        dtype=np.float32,
        chunks=(SHARD_TIME, y_size, x_size),
    )

    template = xr.Dataset(
        data_vars={
            VAR_NAME: xr.DataArray(
                data,
                dims=("time", "y", "x"),
                attrs=first_ds[VAR_NAME].attrs,
            ),
        },
        coords={
            "time": xr.DataArray(
                full_time,
                dims=("time",),
                attrs=first_ds["time"].attrs,
            ),
            "y": first_ds["y"],
            "x": first_ds["x"],
            "lat": first_ds["lat"],
            "lon": first_ds["lon"],
        },
        attrs=dict(first_ds.attrs),
    )

    if "spatial_ref" in first_ds:
        template["spatial_ref"] = first_ds["spatial_ref"]

    if "spatial_ref" in template:
        template["spatial_ref"].attrs.setdefault("epsg_code", epsg_norm)
        template["spatial_ref"].attrs.setdefault(
            "grid_mapping_name",
            "transverse_mercator" if projected else "latitude_longitude",
        )

    return template


def _initialize_store(template: xr.Dataset, output_zarr: Path) -> None:
    y_size = template.sizes["y"]
    x_size = template.sizes["x"]

    encoding = _build_encoding(y_size, x_size)

    _remove_path(output_zarr)
    output_zarr.parent.mkdir(parents=True, exist_ok=True)

    coords_ds = template.drop_vars([VAR_NAME], errors="ignore")

    coords_encoding = {
        key: value
        for key, value in encoding.items()
        if key in coords_ds.variables
    }

    coords_ds.to_zarr(
        str(output_zarr),
        mode="w",
        consolidated=False,
        zarr_format=ZARR_FORMAT,
        encoding=coords_encoding,
    )

    var_only = xr.Dataset(
        data_vars={VAR_NAME: template[VAR_NAME]},
        coords={
            "time": template["time"],
            "y": template["y"],
            "x": template["x"],
        },
        attrs=template.attrs,
    )

    delayed = var_only.to_zarr(
        str(output_zarr),
        mode="a",
        consolidated=False,
        zarr_format=ZARR_FORMAT,
        encoding={VAR_NAME: encoding[VAR_NAME]},
        compute=False,
        align_chunks=True,
        safe_chunks=False,
    )
    del delayed


# =============================================================================
# STEP 3. WRITE BLOCK BY BLOCK
# =============================================================================

def _regularize_month_to_local_full_time(ds: xr.Dataset) -> xr.Dataset:
    ds = _sort_and_drop_duplicate_times(ds)

    original_time = pd.DatetimeIndex(ds["time"].values)
    if len(original_time) == 0:
        return ds

    local_full_time = pd.date_range(
        start=original_time[0],
        end=original_time[-1],
        freq=TIME_FREQ,
    )

    return ds.reindex(time=local_full_time)


def _clip_block_to_global_time(
    block: xr.Dataset,
    full_time: pd.DatetimeIndex,
) -> tuple[xr.Dataset, int, int] | None:
    block_time = pd.DatetimeIndex(block["time"].values)

    mask = block_time.isin(full_time)
    if not mask.any():
        return None

    clipped = block.isel(time=np.where(mask)[0])
    clipped_time = pd.DatetimeIndex(clipped["time"].values)
    gidx = full_time.get_indexer(clipped_time)

    if np.any(gidx < 0):
        raise ValueError("Some times are still not mappable after clipping to the global axis.")

    g0 = int(gidx[0])
    g1 = int(gidx[-1]) + 1

    expected = np.arange(g0, g1)
    if len(expected) != len(gidx) or not np.array_equal(gidx, expected):
        raise ValueError(
            "The clipped block is not contiguous in the global time axis. "
            "Check TIME_FREQ or the temporal regularization."
        )

    return clipped, g0, g1


def _rechunk_block_for_region_write(block_ds: xr.Dataset) -> xr.Dataset:
    y_size = block_ds.sizes["y"]
    x_size = block_ds.sizes["x"]
    t_size = block_ds.sizes["time"]

    rechunk_time = min(SHARD_TIME, t_size)

    out = block_ds.copy()
    out[VAR_NAME] = out[VAR_NAME].chunk(
        {
            "time": rechunk_time,
            "y": y_size,
            "x": x_size,
        }
    )
    return out


def _write_one_month(
    path: Path,
    output_zarr: Path,
    full_time: pd.DatetimeIndex,
    month_idx: int,
    n_months: int,
    epsg: str,
) -> None:
    print(f"\n[write {month_idx}/{n_months}] {path}")

    ds = _open_monthly_zarr(path)
    try:
        ds = _ensure_lat_lon_as_coords(ds)
        ds = _sort_and_drop_duplicate_times(ds)
        ds = _set_cf_attrs(ds, epsg=epsg)
        ds = _clear_inherited_encodings(ds)
        ds = _regularize_month_to_local_full_time(ds)

        if ds.sizes["time"] == 0:
            print("Empty month, skipping.")
            return

        n_time = ds.sizes["time"]
        y_size = ds.sizes["y"]
        x_size = ds.sizes["x"]

        for t0 in range(0, n_time, SHARD_TIME):
            t1 = min(t0 + SHARD_TIME, n_time)

            block = ds.isel(time=slice(t0, t1))

            clipped = _clip_block_to_global_time(block, full_time)
            if clipped is None:
                print(f"  local time block [{t0}:{t1}) is outside the global axis, skipping")
                continue

            block, g0, g1 = clipped
            block_time = pd.DatetimeIndex(block["time"].values)

            print(
                f"  local time block [{t0}:{t1}) -> global [{g0}:{g1}) "
                f"({len(block_time)} steps after clipping)"
            )

            block_ds = xr.Dataset(
                data_vars={
                    VAR_NAME: xr.DataArray(
                        block[VAR_NAME].data.astype(np.float32),
                        dims=("time", "y", "x"),
                        attrs=block[VAR_NAME].attrs,
                    )
                },
                coords={
                    "time": block["time"],
                    "y": block["y"],
                    "x": block["x"],
                },
            )

            block_ds = _rechunk_block_for_region_write(block_ds)

            delayed = block_ds.to_zarr(
                str(output_zarr),
                mode="r+",
                consolidated=False,
                zarr_format=ZARR_FORMAT,
                region={
                    "time": slice(g0, g1),
                    "y": slice(0, y_size),
                    "x": slice(0, x_size),
                },
                compute=False,
                align_chunks=True,
                safe_chunks=False,
            )

            with ProgressBar():
                delayed.compute()

    finally:
        ds.close()


# =============================================================================
# MAIN MERGE FUNCTION
# =============================================================================

def merge_monthly_zarrs_from_directory_by_region(
    input_dir: Path,
    output_zarr: Path,
    epsg: str = DEFAULT_EPSG,
) -> None:
    epsg_norm = normalize_epsg(epsg)

    input_zarrs = _find_input_zarrs(input_dir)

    print("Found input Zarr stores:")
    for z in input_zarrs:
        print(f"  - {z}")

    print(f"\nVariable name: {VAR_NAME}")
    print(f"Dataset CRS:   {epsg_norm}")

    print("\n[1/3] Scanning global times...")
    full_time, union_times = _collect_global_time_info(input_zarrs)

    _report_layout(len(full_time))

    print("\n[2/3] Building global template...")
    first_ds = _open_monthly_zarr(input_zarrs[0])
    try:
        template = _build_template_dataset(first_ds, full_time, epsg=epsg_norm)
    finally:
        first_ds.close()

    print("\nInitializing output store...")
    _initialize_store(template, output_zarr)

    print("\n[3/3] Writing temporal blocks...")
    for i, path in enumerate(input_zarrs, start=1):
        _write_one_month(
            path=path,
            output_zarr=output_zarr,
            full_time=full_time,
            month_idx=i,
            n_months=len(input_zarrs),
            epsg=epsg_norm,
        )

    print(f"\n[OK] Final Zarr store written to: {output_zarr}")


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge monthly radar Zarr stores into a single Zarr store using region writes."
    )

    parser.add_argument(
        "--input-dir",
        "--input_dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing the monthly input Zarr stores.",
    )
    parser.add_argument(
        "--output-zarr",
        "--output_zarr",
        type=str,
        default=str(DEFAULT_OUTPUT_ZARR),
        help="Path to the output merged Zarr store.",
    )
    parser.add_argument(
        "--var-name",
        type=str,
        default=DEFAULT_VAR_NAME,
        help="Name of the radar variable to merge.",
    )
    parser.add_argument(
        "--epsg",
        type=str,
        default=DEFAULT_EPSG,
        help="Dataset CRS. Examples: EPSG:4326, 4326, EPSG:25830 or 25830.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    global VAR_NAME
    global EPSG

    VAR_NAME = args.var_name
    EPSG = normalize_epsg(args.epsg)

    merge_monthly_zarrs_from_directory_by_region(
        input_dir=Path(args.input_dir),
        output_zarr=Path(args.output_zarr),
        epsg=EPSG,
    )


if __name__ == "__main__":
    main()