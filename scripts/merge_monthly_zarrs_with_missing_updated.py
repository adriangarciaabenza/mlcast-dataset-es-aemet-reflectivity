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
DEFAULT_STANDARD_NAME = "equivalent_reflectivity_factor"
DEFAULT_LONG_NAME = "Radar reflectivity"
DEFAULT_UNITS = "dBZ"
DEFAULT_EPSG = "EPSG:25830"

TIME_FREQ = "10min"
TIME_CHUNK = 1
SHARD_TIME = 144
COMPRESSION_LEVEL = 12
ZARR_FORMAT = 3

# Default behaviour:
# - "drop": remove timestamps that are not aligned with TIME_FREQ.
# - "error": fail if any non-aligned timestamp is found.
DEFAULT_NON_ALIGNED_TIME_POLICY = "drop"

VAR_NAME = DEFAULT_VAR_NAME
STANDARD_NAME = DEFAULT_STANDARD_NAME
LONG_NAME = DEFAULT_LONG_NAME
UNITS = DEFAULT_UNITS
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

    if len(time_index) == 0:
        return ds

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
            "long_name": LONG_NAME,
            "standard_name": STANDARD_NAME,
            "units": UNITS,
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


def _build_encoding(y_size: int, x_size: int, n_missing: int = 0) -> dict:
    compressor = ZstdCodec(level=COMPRESSION_LEVEL)

    _validate_chunk_shard_settings()

    encoding = {
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

    if n_missing > 0:
        encoding["missing_times"] = {
            "compressors": (compressor,),
            "chunks": (1,),
        }

    return encoding


# =============================================================================
# TIME HELPERS
# =============================================================================

def _find_sources_for_times(
    zarrs: list[Path],
    target_times: pd.DatetimeIndex,
) -> None:
    """
    Print input Zarr stores containing any target timestamps.
    """
    target_times = pd.DatetimeIndex(target_times).astype("datetime64[ns]")

    if len(target_times) == 0:
        return

    print("\n=== SEARCHING SOURCES FOR SUSPICIOUS TIMES ===")

    for path in zarrs:
        ds = _open_monthly_zarr(path)

        try:
            if "time" not in ds.coords:
                continue

            time_index = pd.DatetimeIndex(ds["time"].values).astype("datetime64[ns]")
            matches = time_index.intersection(target_times)

            if len(matches) > 0:
                print(f"\n{path}")
                print(f"  matches: {matches.tolist()}")

        finally:
            ds.close()


def _get_non_aligned_times(
    observed_time: pd.DatetimeIndex,
    freq: str,
) -> pd.DatetimeIndex:
    """
    Return observed times that are not aligned with the expected frequency grid.

    The first observed timestamp is used as the grid origin.
    """
    observed_time = pd.DatetimeIndex(observed_time).astype("datetime64[ns]")

    if len(observed_time) <= 1:
        return pd.DatetimeIndex([])

    offset = pd.tseries.frequencies.to_offset(freq)
    freq_ns = offset.nanos

    values_ns = observed_time.values.astype("datetime64[ns]").astype("int64")
    origin_ns = observed_time[0].to_datetime64().astype("datetime64[ns]").astype("int64")

    remainder = (values_ns - origin_ns) % freq_ns

    return observed_time[remainder != 0]


def _drop_non_aligned_times_from_dataset(
    ds: xr.Dataset,
    reference_time: pd.DatetimeIndex,
    freq: str,
) -> xr.Dataset:
    """
    Drop timestamps from a dataset that are not aligned with the expected frequency grid.

    The same origin used for the global observed axis is used here.
    """
    ds = ds.copy()

    if "time" not in ds.coords or ds.sizes.get("time", 0) <= 1:
        return ds

    reference_time = pd.DatetimeIndex(reference_time).astype("datetime64[ns]")

    if len(reference_time) == 0:
        return ds

    time_index = pd.DatetimeIndex(ds["time"].values).astype("datetime64[ns]")

    offset = pd.tseries.frequencies.to_offset(freq)
    freq_ns = offset.nanos

    values_ns = time_index.values.astype("datetime64[ns]").astype("int64")
    origin_ns = reference_time[0].to_datetime64().astype("datetime64[ns]").astype("int64")

    aligned_mask = ((values_ns - origin_ns) % freq_ns) == 0

    dropped_times = time_index[~aligned_mask]

    if len(dropped_times) > 0:
        print("\n=== DROPPING NON-ALIGNED TIMES FROM INPUT BLOCK ===")
        print(f"Number of dropped timestamps: {len(dropped_times)}")
        print(f"First dropped timestamps: {dropped_times[:20].tolist()}")

    return ds.isel(time=np.where(aligned_mask)[0])


def _apply_non_aligned_time_policy(
    zarrs: list[Path],
    observed_time: pd.DatetimeIndex,
    non_aligned_time_policy: str,
) -> pd.DatetimeIndex:
    """
    Apply the selected policy to timestamps that are not aligned with TIME_FREQ.
    """
    observed_time = pd.DatetimeIndex(observed_time).astype("datetime64[ns]")

    non_aligned_times = _get_non_aligned_times(
        observed_time=observed_time,
        freq=TIME_FREQ,
    )

    if len(non_aligned_times) == 0:
        return observed_time

    _find_sources_for_times(
        zarrs=zarrs,
        target_times=non_aligned_times,
    )

    if non_aligned_time_policy == "error":
        raise ValueError(
            f"Some observed times are not aligned with the expected {TIME_FREQ} grid.\n"
            f"Number of non-aligned times: {len(non_aligned_times)}\n"
            f"First non-aligned times: {non_aligned_times[:20].tolist()}\n"
            f"First observed time used as grid origin: {observed_time[0]}\n"
            "Use --non-aligned-time-policy drop to discard these timestamps."
        )

    if non_aligned_time_policy == "drop":
        print("\n=== DROPPING NON-ALIGNED TIMES FROM GLOBAL AXIS ===")
        print(f"Number of dropped timestamps: {len(non_aligned_times)}")
        print(f"First dropped timestamps: {non_aligned_times[:20].tolist()}")

        observed_time = observed_time.difference(non_aligned_times)
        return pd.DatetimeIndex(observed_time).astype("datetime64[ns]")

    raise ValueError(
        f"Unknown non_aligned_time_policy: {non_aligned_time_policy!r}. "
        "Allowed values are: 'error', 'drop'."
    )


# =============================================================================
# STEP 1. SCAN GLOBAL TIMES
# =============================================================================

def _collect_global_time_info(
    zarrs: list[Path],
    non_aligned_time_policy: str = DEFAULT_NON_ALIGNED_TIME_POLICY,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Collect observed times and infer missing times.

    The main time coordinate contains only timestamps for which data exists.
    Missing timestamps are stored separately in the missing_times coordinate.

    The regular temporal axis is reconstructed as:

        sorted(time + missing_times)

    Returns
    -------
    observed_time:
        Strictly increasing timestamps for which data is available.

    missing_times:
        Expected timestamps that are missing from the input data.

    regular_full_time:
        Complete expected regular temporal axis.
    """
    observed_values: list[np.ndarray] = []

    for i, path in enumerate(zarrs, start=1):
        print(f"[scan {i}/{len(zarrs)}] {path}")

        ds = _open_monthly_zarr(path)

        try:
            ds = _sort_and_drop_duplicate_times(ds)
            time_index = pd.DatetimeIndex(ds["time"].values).astype("datetime64[ns]")

            if len(time_index) == 0:
                continue

            observed_values.append(time_index.values)

        finally:
            ds.close()

    if not observed_values:
        raise ValueError("No valid times were found in the input Zarr stores.")

    observed_time = pd.DatetimeIndex(
        np.sort(np.unique(np.concatenate(observed_values)))
    ).astype("datetime64[ns]")

    if observed_time.has_duplicates:
        raise ValueError("Observed time values contain duplicates.")

    if not observed_time.is_monotonic_increasing:
        raise ValueError("Observed time values are not strictly increasing.")

    observed_time = _apply_non_aligned_time_policy(
        zarrs=zarrs,
        observed_time=observed_time,
        non_aligned_time_policy=non_aligned_time_policy,
    )

    if len(observed_time) == 0:
        raise ValueError("All observed timestamps were removed by the time policy.")

    regular_full_time = pd.date_range(
        start=observed_time[0],
        end=observed_time[-1],
        freq=TIME_FREQ,
    ).astype("datetime64[ns]")

    extra_observed_times = observed_time.difference(regular_full_time)

    if len(extra_observed_times) > 0:
        _find_sources_for_times(
            zarrs=zarrs,
            target_times=extra_observed_times,
        )

        raise ValueError(
            "Some observed timestamps are outside the expected regular "
            f"{TIME_FREQ} temporal axis.\n"
            f"Number of extra observed timestamps: {len(extra_observed_times)}\n"
            f"First extra observed timestamps: {extra_observed_times[:20].tolist()}\n"
            f"First observed time: {observed_time[0]}\n"
            f"Last observed time: {observed_time[-1]}\n"
            f"First regular time: {regular_full_time[0]}\n"
            f"Last regular time: {regular_full_time[-1]}"
        )

    missing_times = regular_full_time.difference(observed_time)
    missing_times = pd.DatetimeIndex(missing_times).astype("datetime64[ns]")

    print("\n=== TIME AXIS ===")
    print(f"regular expected timesteps : {len(regular_full_time)}")
    print(f"observed timesteps         : {len(observed_time)}")
    print(f"missing timesteps          : {len(missing_times)}")
    print(f"first observed time        : {observed_time[0]}")
    print(f"last observed time         : {observed_time[-1]}")

    if len(missing_times) > 0:
        print(f"first missing time         : {missing_times[0]}")
        print(f"last missing time          : {missing_times[-1]}")

    return observed_time, missing_times, regular_full_time


def _validate_time_requirements(
    observed_time: pd.DatetimeIndex,
    missing_times: pd.DatetimeIndex,
    regular_full_time: pd.DatetimeIndex,
) -> None:
    """
    Validate temporal requirements.

    Requirements:
    - The main time coordinate contains only observed timestamps.
    - missing_times contains expected timestamps for which data is missing.
    - missing_times values are not included in the main time coordinate.
    - Time values are strictly monotonically increasing.
    - observed_time + missing_times reconstructs the complete regular axis.
    """
    observed_time = pd.DatetimeIndex(observed_time).astype("datetime64[ns]")
    missing_times = pd.DatetimeIndex(missing_times).astype("datetime64[ns]")
    regular_full_time = pd.DatetimeIndex(regular_full_time).astype("datetime64[ns]")

    if len(observed_time) == 0:
        raise ValueError("The main time coordinate would be empty.")

    if observed_time.has_duplicates:
        duplicated = observed_time[observed_time.duplicated()]
        raise ValueError(
            "The main time coordinate contains duplicates. "
            f"First duplicated values: {duplicated[:10].tolist()}"
        )

    if not observed_time.is_monotonic_increasing:
        raise ValueError("The main time coordinate is not strictly increasing.")

    if missing_times.has_duplicates:
        duplicated = missing_times[missing_times.duplicated()]
        raise ValueError(
            "The missing_times coordinate contains duplicates. "
            f"First duplicated values: {duplicated[:10].tolist()}"
        )

    if len(missing_times) > 0 and not missing_times.is_monotonic_increasing:
        raise ValueError("The missing_times coordinate is not strictly increasing.")

    overlap = np.intersect1d(
        observed_time.values.astype("datetime64[ns]"),
        missing_times.values.astype("datetime64[ns]"),
    )

    if len(overlap) > 0:
        raise ValueError(
            "Some timestamps are present both in time and missing_times. "
            "Missing times must not be included in the main time coordinate. "
            f"First overlapping values: {pd.DatetimeIndex(overlap[:10]).tolist()}"
        )

    combined = pd.DatetimeIndex(
        np.sort(
            np.concatenate(
                [
                    observed_time.values.astype("datetime64[ns]"),
                    missing_times.values.astype("datetime64[ns]"),
                ]
            )
        )
    )

    if combined.has_duplicates:
        duplicated = combined[combined.duplicated()]
        raise ValueError(
            "Combined observed and missing times contain duplicates. "
            f"First duplicated values: {duplicated[:10].tolist()}"
        )

    if not combined.is_monotonic_increasing:
        raise ValueError("Combined observed and missing times are not strictly increasing.")

    if len(combined) != len(regular_full_time):
        missing_from_combined = regular_full_time.difference(combined)
        extra_in_combined = combined.difference(regular_full_time)

        raise ValueError(
            "Observed time plus missing_times does not reconstruct the expected "
            f"regular {TIME_FREQ} temporal axis because lengths differ.\n"
            f"Expected length: {len(regular_full_time)}\n"
            f"Combined length: {len(combined)}\n"
            f"Number of values missing from combined: {len(missing_from_combined)}\n"
            f"Number of extra values in combined: {len(extra_in_combined)}\n"
            f"First values missing from combined: {missing_from_combined[:20].tolist()}\n"
            f"First extra values in combined: {extra_in_combined[:20].tolist()}\n"
            f"First combined time: {combined[0]}\n"
            f"Last combined time: {combined[-1]}\n"
            f"First regular time: {regular_full_time[0]}\n"
            f"Last regular time: {regular_full_time[-1]}"
        )

    mismatch_mask = combined.values != regular_full_time.values

    if np.any(mismatch_mask):
        mismatch_idx = np.where(mismatch_mask)[0][:20]

        details = []
        for idx in mismatch_idx:
            details.append(
                {
                    "index": int(idx),
                    "expected": str(regular_full_time[idx]),
                    "actual": str(combined[idx]),
                }
            )

        raise ValueError(
            "Observed time plus missing_times does not reconstruct the expected "
            f"regular {TIME_FREQ} temporal axis.\n"
            f"First mismatches: {details}"
        )


# =============================================================================
# STEP 2. CREATE GLOBAL TEMPLATE
# =============================================================================

def _build_template_dataset(
    first_ds: xr.Dataset,
    observed_time: pd.DatetimeIndex,
    missing_times: pd.DatetimeIndex,
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
        (len(observed_time), y_size, x_size),
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
                observed_time.values.astype("datetime64[ns]"),
                dims=("time",),
                attrs=first_ds["time"].attrs,
            ),
            "y": first_ds["y"],
            "x": first_ds["x"],
            "lat": first_ds["lat"],
            "lon": first_ds["lon"],
            "missing_times": xr.DataArray(
                missing_times.values.astype("datetime64[ns]"),
                dims=("missing_times",),
                attrs={
                    "standard_name": "time",
                    "long_name": "missing times",
                    "description": (
                        "Timestamps expected in the regular temporal archive "
                        "but not present in the main time coordinate because data is missing."
                    ),
                },
            ),
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

    template.attrs["consistent_timestep_start"] = (
        pd.Timestamp(observed_time[0]).strftime("%Y-%m-%dT%H:%M")
    )

    template.attrs["base_frequencies"] = (
        f"{TIME_FREQ}:{pd.Timestamp(observed_time[0]).strftime('%Y-%m-%dT%H:%M')}/None"
    )

    return template


def _initialize_store(template: xr.Dataset, output_zarr: Path) -> None:
    y_size = template.sizes["y"]
    x_size = template.sizes["x"]
    n_missing = template.sizes.get("missing_times", 0)

    encoding = _build_encoding(
        y_size=y_size,
        x_size=x_size,
        n_missing=n_missing,
    )

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
        data_vars={
            VAR_NAME: template[VAR_NAME],
        },
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

def _clip_block_to_global_time(
    block: xr.Dataset,
    observed_time: pd.DatetimeIndex,
) -> tuple[xr.Dataset, int, int] | None:
    block_time = pd.DatetimeIndex(block["time"].values).astype("datetime64[ns]")
    observed_time = pd.DatetimeIndex(observed_time).astype("datetime64[ns]")

    mask = block_time.isin(observed_time)

    if not mask.any():
        return None

    clipped = block.isel(time=np.where(mask)[0])
    clipped_time = pd.DatetimeIndex(clipped["time"].values).astype("datetime64[ns]")

    global_indices = observed_time.get_indexer(clipped_time)

    if np.any(global_indices < 0):
        raise ValueError(
            "Some times are still not mappable after clipping to the observed time axis."
        )

    g0 = int(global_indices[0])
    g1 = int(global_indices[-1]) + 1

    expected = np.arange(g0, g1)

    if len(expected) != len(global_indices) or not np.array_equal(
        global_indices,
        expected,
    ):
        raise ValueError(
            "The clipped block is not contiguous in the observed time axis. "
            "This usually indicates duplicated, unsorted or inconsistent timestamps."
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
    observed_time: pd.DatetimeIndex,
    month_idx: int,
    n_months: int,
    epsg: str,
    non_aligned_time_policy: str = DEFAULT_NON_ALIGNED_TIME_POLICY,
) -> None:
    print(f"\n[write {month_idx}/{n_months}] {path}")

    ds = _open_monthly_zarr(path)

    try:
        ds = _ensure_lat_lon_as_coords(ds)
        ds = _sort_and_drop_duplicate_times(ds)
        ds = _set_cf_attrs(ds, epsg=epsg)
        ds = _clear_inherited_encodings(ds)

        if non_aligned_time_policy == "drop":
            ds = _drop_non_aligned_times_from_dataset(
                ds=ds,
                reference_time=observed_time,
                freq=TIME_FREQ,
            )

        if ds.sizes["time"] == 0:
            print("Empty month after temporal filtering, skipping.")
            return

        n_time = ds.sizes["time"]
        y_size = ds.sizes["y"]
        x_size = ds.sizes["x"]

        for t0 in range(0, n_time, SHARD_TIME):
            t1 = min(t0 + SHARD_TIME, n_time)

            block = ds.isel(time=slice(t0, t1))

            clipped = _clip_block_to_global_time(
                block=block,
                observed_time=observed_time,
            )

            if clipped is None:
                print(
                    f"  local time block [{t0}:{t1}) is outside the observed axis, "
                    "skipping"
                )
                continue

            block, g0, g1 = clipped
            block_time = pd.DatetimeIndex(block["time"].values)

            print(
                f"  local time block [{t0}:{t1}) -> global [{g0}:{g1}) "
                f"({len(block_time)} observed steps)"
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
    non_aligned_time_policy: str = DEFAULT_NON_ALIGNED_TIME_POLICY,
) -> None:
    epsg_norm = normalize_epsg(epsg)

    input_zarrs = _find_input_zarrs(input_dir)

    print("Found input Zarr stores:")
    for zarr_path in input_zarrs:
        print(f"  - {zarr_path}")

    print(f"\nVariable name: {VAR_NAME}")
    print(f"Standard name: {STANDARD_NAME}")
    print(f"Long name:     {LONG_NAME}")
    print(f"Units:         {UNITS}")
    print(f"Dataset CRS:   {epsg_norm}")
    print(f"Non-aligned time policy: {non_aligned_time_policy}")

    print("\n[1/3] Scanning global times...")

    observed_time, missing_times, regular_full_time = _collect_global_time_info(
        zarrs=input_zarrs,
        non_aligned_time_policy=non_aligned_time_policy,
    )

    _validate_time_requirements(
        observed_time=observed_time,
        missing_times=missing_times,
        regular_full_time=regular_full_time,
    )

    _report_layout(len(observed_time))

    print("\n[2/3] Building global template...")

    first_ds = _open_monthly_zarr(input_zarrs[0])

    try:
        template = _build_template_dataset(
            first_ds=first_ds,
            observed_time=observed_time,
            missing_times=missing_times,
            epsg=epsg_norm,
        )

    finally:
        first_ds.close()

    print("\nInitializing output store...")
    _initialize_store(template, output_zarr)

    print("\n[3/3] Writing temporal blocks...")

    for i, path in enumerate(input_zarrs, start=1):
        _write_one_month(
            path=path,
            output_zarr=output_zarr,
            observed_time=observed_time,
            month_idx=i,
            n_months=len(input_zarrs),
            epsg=epsg_norm,
            non_aligned_time_policy=non_aligned_time_policy,
        )

    print(f"\n[OK] Final Zarr store written to: {output_zarr}")


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge monthly radar Zarr stores into a single Zarr store using "
            "region writes. Missing timestamps are stored explicitly in the "
            "missing_times coordinate and are not included in the main time coordinate."
        )
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
        "--standard-name",
        type=str,
        default=DEFAULT_STANDARD_NAME,
        help="CF standard_name for the main data variable.",
    )

    parser.add_argument(
        "--long-name",
        type=str,
        default=DEFAULT_LONG_NAME,
        help="Long name for the main data variable.",
    )

    parser.add_argument(
        "--units",
        type=str,
        default=DEFAULT_UNITS,
        help="Units for the main data variable.",
    )

    parser.add_argument(
        "--epsg",
        type=str,
        default=DEFAULT_EPSG,
        help="Dataset CRS. Examples: EPSG:4326, 4326, EPSG:25830 or 25830.",
    )

    parser.add_argument(
        "--non-aligned-time-policy",
        choices=["error", "drop"],
        default=DEFAULT_NON_ALIGNED_TIME_POLICY,
        help=(
            "Policy for observed timestamps that are not aligned with TIME_FREQ. "
            "'drop' removes those timestamps from the merge; "
            "'error' fails explicitly. Default: drop."
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    global VAR_NAME
    global STANDARD_NAME
    global LONG_NAME
    global UNITS
    global EPSG

    VAR_NAME = args.var_name
    STANDARD_NAME = args.standard_name
    LONG_NAME = args.long_name
    UNITS = args.units
    EPSG = normalize_epsg(args.epsg)

    merge_monthly_zarrs_from_directory_by_region(
        input_dir=Path(args.input_dir),
        output_zarr=Path(args.output_zarr),
        epsg=EPSG,
        non_aligned_time_policy=args.non_aligned_time_policy,
    )


if __name__ == "__main__":
    main()