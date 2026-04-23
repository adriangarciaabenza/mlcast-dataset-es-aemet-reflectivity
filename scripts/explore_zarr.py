from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from mlcast_dataset_es_aemet_reflectivity.radar_inspection import (
    inspect_radar_dataset_in_memory,
)

# =============================================================================
# DEFAULT CONFIG
# =============================================================================

DEFAULT_ZARR_PATH = Path("./ES-AEMET-radar_reflectivity-ppi_ZAR_2020-2024.zarr/")
DEFAULT_VAR_NAME = "equivalent_reflectivity_factor"
DEFAULT_TIME_INDEX = 0
DEFAULT_USE_SEL = False
DEFAULT_TIME_VALUE = None


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


# =============================================================================
# MAIN LOGIC
# =============================================================================


def run(
    zarr_path: Path,
    var_name: str,
    time_index: int,
    use_sel: bool,
    time_value: str | None,
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
        # print(f"{v}: shape={ds[v].shape}, dtype={ds[v].dtype}, chunks={chunks}")

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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    zarr_path = Path(args.zarr_path) if args.zarr_path else DEFAULT_ZARR_PATH
    var_name = args.var_name or DEFAULT_VAR_NAME
    time_index = args.time_index if args.time_index is not None else DEFAULT_TIME_INDEX
    use_sel = args.use_sel if args.use_sel is not None else DEFAULT_USE_SEL
    time_value = args.time_value if args.time_value is not None else DEFAULT_TIME_VALUE

    run(
        zarr_path=zarr_path,
        var_name=var_name,
        time_index=time_index,
        use_sel=use_sel,
        time_value=time_value,
    )


if __name__ == "__main__":
    main()
