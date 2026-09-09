from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from mlcast_dataset_es_aemet_reflectivity.radar_inspection import (
    inspect_radar_dataset_in_memory,
)

# =============================================================================
# DEFAULT CONFIG
# =============================================================================

#DEFAULT_ZARR_PATH = Path("/lustre/utmp/std/MLCAST_radar_data/outputs/monthly_SRI_v1/radar_2020_01.zarr")
DEFAULT_ZARR_PATH = Path("/lustre/utmp/std/MLCAST_radar_data/ES-AEMET-radar_reflectivity-ppi_ZAR_2020-2024_v3.zarr")
DEFAULT_VAR_NAME = "dbz"
DEFAULT_TIME_INDEX = 0
DEFAULT_USE_SEL = False
DEFAULT_TIME_VALUE = None
DEFAULT_EXPECTED_FREQ = "10min"
DEFAULT_PLOT_MISSING_TIMES = False


# =============================================================================
# HELPERS
# =============================================================================


def detect_zarr_format(path: Path) -> int:
    """
    Detect the Zarr format in a simple way:
    - v3 if zarr.json exists at the root
    - v2 otherwise
    """
    if (path / "zarr.json").exists():
        return 3
    return 2


def open_zarr_dataset(path: Path) -> xr.Dataset:
    zarr_format = detect_zarr_format(path)

    print("\n=== OPENING ZARR ===")
    print(f"Path: {path.resolve()}")
    print(f"Exists: {path.exists()}")
    print(f"Detected Zarr format: v{zarr_format}")

    if not path.exists():
        raise FileNotFoundError(f"Zarr store does not exist: {path}")

    if zarr_format == 3:
        ds = xr.open_zarr(path, consolidated=None, zarr_format=3)
    else:
        ds = xr.open_zarr(path, consolidated=True)

    return ds


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _as_datetime_index(ds: xr.Dataset, coord_name: str) -> pd.DatetimeIndex:
    if coord_name not in ds.coords:
        raise KeyError(f"Coordinate '{coord_name}' does not exist in the dataset")

    return pd.DatetimeIndex(ds[coord_name].values).astype("datetime64[ns]")


def _print_datetime_index_summary(
    name: str,
    index: pd.DatetimeIndex,
    n_preview: int = 10,
) -> None:
    print(f"\n=== {name.upper()} SUMMARY ===")
    print(f"size              : {len(index)}")

    if len(index) == 0:
        print("empty             : yes")
        return

    print(f"dtype             : {index.dtype}")
    print(f"first             : {index[0]}")
    print(f"last              : {index[-1]}")
    print(f"has duplicates    : {index.has_duplicates}")
    print(f"is monotonic inc. : {index.is_monotonic_increasing}")

    print(f"\nFirst {min(n_preview, len(index))} values:")
    for value in index[:n_preview]:
        print(f"  - {value}")

    print(f"\nLast {min(n_preview, len(index))} values:")
    for value in index[-n_preview:]:
        print(f"  - {value}")


def inspect_missing_times(
    ds: xr.Dataset,
    expected_freq: str = DEFAULT_EXPECTED_FREQ,
    plot_missing_times: bool = DEFAULT_PLOT_MISSING_TIMES,
) -> None:
    """
    Inspect the missing_times coordinate.

    Expected MLCAST-like structure:

    Dimensions:
        time
        y
        x
        missing_times

    Coordinates:
      * time           (time) datetime64[ns]
      * missing_times  (missing_times) datetime64[ns]

    The complete expected temporal axis should be reconstructed as:

        sorted(time ∪ missing_times)
    """
    print("\n" + "=" * 80)
    print("MISSING TIMES INSPECTION")
    print("=" * 80)

    if "time" not in ds.coords:
        print("No 'time' coordinate found. Skipping missing_times inspection.")
        return

    time_index = _as_datetime_index(ds, "time")
    _print_datetime_index_summary("time", time_index)

    if "missing_times" not in ds.coords:
        print("\nNo 'missing_times' coordinate found.")
        return

    missing_index = _as_datetime_index(ds, "missing_times")
    _print_datetime_index_summary("missing_times", missing_index)

    print("\n=== MISSING_TIMES COORD DETAILS ===")
    print(ds["missing_times"])
    print("\nattrs:")
    print(ds["missing_times"].attrs)
    print("\nencoding:")
    print(ds["missing_times"].encoding)
    print(f"\nchunks: {getattr(ds['missing_times'], 'chunks', None)}")

    print("\n=== TEMPORAL CONSISTENCY CHECKS ===")

    overlap = time_index.intersection(missing_index)
    print(f"time ∩ missing_times size: {len(overlap)}")
    if len(overlap) > 0:
        print("First overlapping values:")
        for value in overlap[:20]:
            print(f"  - {value}")

    combined = pd.DatetimeIndex(
        sorted(time_index.union(missing_index))
    ).astype("datetime64[ns]")

    _print_datetime_index_summary("combined time + missing_times", combined)

    if len(combined) > 1:
        inferred_diffs = combined.to_series().diff().dropna()
        diff_counts = inferred_diffs.value_counts().sort_index()

        print("\n=== COMBINED TEMPORAL DIFFERENCES ===")
        print(diff_counts.head(20))

        expected_delta = pd.Timedelta(expected_freq)
        non_regular_steps = inferred_diffs[inferred_diffs != expected_delta]

        print(f"\nExpected frequency       : {expected_freq}")
        print(f"Expected delta           : {expected_delta}")
        print(f"Non-regular step count   : {len(non_regular_steps)}")

        if len(non_regular_steps) > 0:
            print("\nFirst non-regular steps:")
            for timestamp, delta in non_regular_steps.head(20).items():
                previous_timestamp = combined[combined.get_loc(timestamp) - 1]
                print(
                    f"  - previous={previous_timestamp}, "
                    f"current={timestamp}, delta={delta}"
                )

    if len(combined) > 0:
        expected_full_time = pd.date_range(
            start=combined[0],
            end=combined[-1],
            freq=expected_freq,
        ).astype("datetime64[ns]")

        missing_from_reconstructed = expected_full_time.difference(combined)
        extra_in_reconstructed = combined.difference(expected_full_time)

        print("\n=== RECONSTRUCTION AGAINST EXPECTED FULL AXIS ===")
        print(f"expected full axis size      : {len(expected_full_time)}")
        print(f"combined reconstructed size  : {len(combined)}")
        print(f"missing from reconstructed   : {len(missing_from_reconstructed)}")
        print(f"extra in reconstructed       : {len(extra_in_reconstructed)}")

        if len(missing_from_reconstructed) > 0:
            print("\nFirst timestamps missing from reconstructed axis:")
            for value in missing_from_reconstructed[:20]:
                print(f"  - {value}")

        if len(extra_in_reconstructed) > 0:
            print("\nFirst extra timestamps in reconstructed axis:")
            for value in extra_in_reconstructed[:20]:
                print(f"  - {value}")

    print("\n=== MISSING TIMES BY YEAR ===")
    if len(missing_index) > 0:
        missing_by_year = pd.Series(1, index=missing_index).groupby(
            missing_index.year
        ).sum()
        print(missing_by_year.to_string())
    else:
        print("No missing times.")

    print("\n=== MISSING TIMES BY MONTH ===")
    if len(missing_index) > 0:
        missing_by_month = pd.Series(1, index=missing_index).groupby(
            missing_index.to_period("M")
        ).sum()
        print(missing_by_month.to_string())
    else:
        print("No missing times.")

    print("\n=== GLOBAL ATTRIBUTES RELATED TO TIMESTEP ===")
    for attr_name in ["consistent_timestep_start", "base_frequencies"]:
        print(f"{attr_name}: {ds.attrs.get(attr_name, None)}")

    if plot_missing_times and len(missing_index) > 0:
        plot_missing_times_distribution(missing_index)


def plot_missing_times_distribution(missing_index: pd.DatetimeIndex) -> None:
    """
    Plot the number of missing times per month.
    """
    missing_by_month = pd.Series(1, index=missing_index).groupby(
        missing_index.to_period("M")
    ).sum()

    missing_by_month.index = missing_by_month.index.to_timestamp()

    plt.figure(figsize=(12, 5))
    plt.plot(missing_by_month.index, missing_by_month.values, marker="o")
    plt.xlabel("Month")
    plt.ylabel("Number of missing timesteps")
    plt.title("Missing timesteps per month")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN LOGIC
# =============================================================================


def run(
    zarr_path: Path,
    var_name: str,
    time_index: int,
    use_sel: bool,
    time_value: str | None,
    expected_freq: str,
    plot_missing_times: bool,
) -> None:
    ds = open_zarr_dataset(zarr_path)

    print("\n=== DATASET ===")
    print(ds)

    print("\n=== DIMS ===")
    print(ds.dims)

    print("\n=== COORDS ===")
    for c in ds.coords:
        print(f"{c}: shape={ds[c].shape}, dtype={ds[c].dtype}")

    print("\n=== DATA VARS ===")
    for v in ds.data_vars:
        chunks = getattr(ds[v], "chunks", None)
        print(ds[v].attrs)
        #print(f"{v}: shape={ds[v].shape}, dtype={ds[v].dtype}, chunks={chunks}")

    inspect_missing_times(
        ds=ds,
        expected_freq=expected_freq,
        plot_missing_times=plot_missing_times,
    )

    if var_name not in ds:
        raise KeyError(f"Variable '{var_name}' does not exist in the dataset")

    da = ds[var_name]

    print("\n=== SELECTED VARIABLE ===")
    print(da)
    print("\n=== ATTRS ===")
    print(da.attrs)
    print("\n=== ENCODING ===")
    print(da.encoding)

    # Quick inspection
    if da.sizes.get("time", 0) > 0:
        print("\n=== QUICK IN-MEMORY INSPECTION ===")
        inspect_radar_dataset_in_memory(ds, var_name=var_name, time_index=time_index)

    # Select 2D field
    print("\n=== SELECTING 2D FIELD ===")
    if use_sel:
        if time_value is None:
            raise ValueError("If use_sel=True, time_value cannot be None")
        field = da.sel(time=time_value)
    else:
        field = da.isel(time=time_index)

    print(field)

    print("\n=== LOADING ONLY THIS 2D FIELD ===")
    field_loaded = field.compute()
    print(field_loaded)

    # Plot
    plt.figure(figsize=(10, 6))
    if (
        "lat" in ds.coords
        and "lon" in ds.coords
        and ds["lat"].ndim == 2
        and ds["lon"].ndim == 2
    ):
        plt.pcolormesh(
            ds["lon"].values,
            ds["lat"].values,
            field_loaded.values,
            shading="auto",
        )
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.colorbar(label=field_loaded.attrs.get("units", ""))
    else:
        field_loaded.plot(cmap="viridis")

    title_time = getattr(field_loaded, "time", None)
    if title_time is not None:
        plt.title(f"{var_name} | time={field_loaded.time.values}")
    else:
        plt.title(var_name)

    plt.tight_layout()
    plt.show()

    ds.close()


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and visualize a radar Zarr dataset"
    )

    parser.add_argument("--zarr_path", type=str, default=None)
    parser.add_argument("--var_name", type=str, default=None)
    parser.add_argument("--time_index", type=int, default=None)

    parser.add_argument("--use_sel", type=str2bool, default=None)
    parser.add_argument("--time_value", type=str, default=None)

    parser.add_argument(
        "--expected_freq",
        type=str,
        default=DEFAULT_EXPECTED_FREQ,
        help="Expected temporal frequency used to validate time + missing_times.",
    )

    parser.add_argument(
        "--plot_missing_times",
        type=str2bool,
        default=DEFAULT_PLOT_MISSING_TIMES,
        help="Whether to plot the number of missing timesteps per month.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    zarr_path = Path(args.zarr_path) if args.zarr_path else DEFAULT_ZARR_PATH
    var_name = args.var_name or DEFAULT_VAR_NAME
    time_index = args.time_index if args.time_index is not None else DEFAULT_TIME_INDEX
    use_sel = args.use_sel if args.use_sel is not None else DEFAULT_USE_SEL
    time_value = args.time_value if args.time_value is not None else DEFAULT_TIME_VALUE

    expected_freq = args.expected_freq
    plot_missing_times = args.plot_missing_times

    run(
        zarr_path=zarr_path,
        var_name=var_name,
        time_index=time_index,
        use_sel=use_sel,
        time_value=time_value,
        expected_freq=expected_freq,
        plot_missing_times=plot_missing_times,
    )


if __name__ == "__main__":
    main()