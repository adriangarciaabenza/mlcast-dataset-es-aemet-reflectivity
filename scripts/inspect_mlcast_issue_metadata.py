from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


#DEFAULT_ZARR_PATH = Path("/lustre/utmp/std/MLCAST_radar_data/ES-AEMET-rainfall_rate-sri_ZAR_2020-2024_v1_kg.zarr")
DEFAULT_ZARR_PATH = Path("/lustre/utmp/std/MLCAST_radar_data/ES-AEMET-radar_reflectivity-ppi_ZAR_2020-2024_v3.zarr")
DEFAULT_VAR_NAME = "equivalent_reflectivity_factor"


def detect_zarr_format(path: Path) -> int:
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
        return xr.open_zarr(path, consolidated=True, zarr_format=3)

    return xr.open_zarr(path, consolidated=True)


def infer_temporal_resolution(time_values: np.ndarray) -> str:
    if len(time_values) < 2:
        return "Unknown"

    times = pd.to_datetime(time_values)
    diffs = times.to_series().diff().dropna()

    if diffs.empty:
        return "Unknown"

    most_common = diffs.mode().iloc[0]
    return str(most_common)


def infer_update_frequency(time_values: np.ndarray) -> str:
    if len(time_values) < 2:
        return "Unknown"

    temporal_resolution = infer_temporal_resolution(time_values)

    if "0 days 00:10:00" in temporal_resolution:
        return "Every 10 minutes / Periodic"
    if "0 days 00:05:00" in temporal_resolution:
        return "Every 5 minutes / Periodic"
    if "0 days 01:00:00" in temporal_resolution:
        return "Hourly / Periodic"
    if "1 days" in temporal_resolution:
        return "Daily / Periodic"

    return "Periodic"


def infer_spatial_resolution(ds: xr.Dataset) -> str:
    if "lat" not in ds.coords or "lon" not in ds.coords:
        return "Unknown"

    lat = ds["lat"].values
    lon = ds["lon"].values

    if lat.ndim != 2 or lon.ndim != 2:
        return "Unknown"

    mid_y = lat.shape[0] // 2
    mid_x = lon.shape[1] // 2

    dlat = np.nanmedian(np.abs(np.diff(lat[:, mid_x])))
    dlon = np.nanmedian(np.abs(np.diff(lon[mid_y, :])))

    return f"Approx. {dlat:.5f}° latitude x {dlon:.5f}° longitude"


def infer_bbox(ds: xr.Dataset) -> dict[str, float | str]:
    if "lat" not in ds.coords or "lon" not in ds.coords:
        return {
            "north": "Unknown",
            "east": "Unknown",
            "south": "Unknown",
            "west": "Unknown",
        }

    lat = ds["lat"].values
    lon = ds["lon"].values

    return {
        "north": float(np.nanmax(lat)),
        "east": float(np.nanmax(lon)),
        "south": float(np.nanmin(lat)),
        "west": float(np.nanmin(lon)),
    }


def infer_crs(ds: xr.Dataset, da: xr.DataArray) -> str:
    grid_mapping_name = da.attrs.get("grid_mapping")

    if grid_mapping_name and grid_mapping_name in ds:
        spatial_ref = ds[grid_mapping_name]
        epsg = spatial_ref.attrs.get("epsg_code")
        crs_wkt = spatial_ref.attrs.get("crs_wkt")
        spatial_ref_attr = spatial_ref.attrs.get("spatial_ref")

        if epsg:
            return f"EPSG:{epsg}"
        if crs_wkt:
            return "CRS WKT available in spatial_ref variable"
        if spatial_ref_attr:
            return "Spatial reference available in spatial_ref variable"

    if "spatial_ref" in ds:
        attrs = ds["spatial_ref"].attrs
        epsg = attrs.get("epsg_code")
        if epsg:
            return f"EPSG:{epsg}"
        return "spatial_ref variable available"

    return "Unknown"


def infer_dimensions(da: xr.DataArray) -> str:
    spatial_dims = [dim for dim in da.dims if dim != "time"]

    if len(spatial_dims) >= 2:
        y_dim, x_dim = spatial_dims[-2], spatial_dims[-1]
        return f"{da.sizes[y_dim]} x {da.sizes[x_dim]} pixels"

    return str(dict(da.sizes))


def print_attrs(title: str, attrs: dict) -> None:
    print(f"\n=== {title} ===")
    if not attrs:
        print("No attributes found.")
        return

    for key, value in attrs.items():
        print(f"{key}: {value}")


def build_issue_markdown(
    ds: xr.Dataset,
    da: xr.DataArray,
    zarr_path: Path,
    var_name: str,
) -> str:
    time_values = ds["time"].values if "time" in ds.coords else np.array([])

    if len(time_values) > 0:
        start_time = str(pd.to_datetime(time_values[0]).date())
        end_time = str(pd.to_datetime(time_values[-1]).date())
        temporal_resolution = infer_temporal_resolution(time_values)
        update_frequency = infer_update_frequency(time_values)
    else:
        start_time = "Unknown"
        end_time = "Unknown"
        temporal_resolution = "Unknown"
        update_frequency = "Unknown"

    bbox = infer_bbox(ds)
    spatial_resolution = infer_spatial_resolution(ds)
    dimensions = infer_dimensions(da)
    crs = infer_crs(ds, da)

    units = da.attrs.get("units", "Unknown")
    standard_name = da.attrs.get("standard_name", var_name)
    long_name = da.attrs.get("long_name", standard_name)

    zarr_format = detect_zarr_format(zarr_path)

    variable_label = "Reflectivity (dBZ)"
    if units not in {"dBZ", "dBz", "dbz"}:
        variable_label = f"{long_name} ({units})"

    global_license = ds.attrs.get("license", "CC-BY-4.0")
    institution = ds.attrs.get("institution", "Agencia Estatal de Meteorología (AEMET)")
    source = ds.attrs.get("source", "AEMET weather radar data")
    lineage = ds.attrs.get(
        "data_lineage",
        "Original AEMET radar files converted to CF-compliant Zarr format for MLCAST.",
    )

    return f"""# `ES - AEMET`

## Short description

Spanish AEMET radar reflectivity dataset converted to Zarr format. The dataset provides radar reflectivity fields over Spain with a temporal resolution of {temporal_resolution}, spatial dimensions of {dimensions}, and geographical coordinates stored as latitude/longitude grids.

## 🗂️ Metadata

- **📛 Dataset Name:** `ES-AEMET-radar_reflectivity-ppi_ZAR`
- **⏰ Last Update:** `{date.today().isoformat()}`
- **📅 Update Frequency:** `{update_frequency}`
- **📏 Spatial Resolution:** `{spatial_resolution}`
- **⏱️ Temporal Resolution (Timestep):** `{temporal_resolution}`
- **📐 Dimensions:** `{dimensions}`
- **🗄️ File Format (example filename):** `Zarr v{zarr_format} ({zarr_path.name})`
- **🗜️ Archive structure (example filename):** `One Zarr store containing the full temporal archive`
- **🔑 License:** `[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)`
- **🌍 Country Coverage:** `Spain`
- **🌐 Geographical Coverage:**
  - **📍 North Bound Latitude:** `{bbox["north"]}`
  - **📍 East Bound Longitude:** `{bbox["east"]}`
  - **📍 South Bound Latitude:** `{bbox["south"]}`
  - **📍 West Bound Longitude:** `{bbox["west"]}`
- **📊 Variables (Units):**
  - `{variable_label}`
- **📬 Maintainer:** `{institution}`
- **🕒 Temporal Coverage:**
  - **⏳ Start Time:** `{start_time}`
  - **⏳ End Time:** `{end_time}`
- **🧭 Reference System Identifier:** `{crs}`
- **🔗 Identifier:** `N/A`
- **📜 Data lineage:** `{lineage}`
- **⚠️ Use Limitation:** `Subject to the original AEMET data access conditions and attribution requirements.`

## 🔗 Resources

- **📥 Download URL:** `[AEMET Big Data Platform](http://bigdata.aemet.es/bigdata/inicio)` *(restricted access, registration required)*
- **📄 License URL:** `[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)`
- **📄 Other URL:** `[AEMET OpenData](https://opendata.aemet.es/)` *(public API, does not include radar reflectivity products)*

## 📚 Supplemental Information

- 🌐 `[AEMET Big Data Portal](http://bigdata.aemet.es/bigdata/inicio)` *(main access point for radar data, restricted)*
- 🌐 `[AEMET OpenData](https://opendata.aemet.es/)` *(public datasets and API)*
"""


def inspect_dataset(zarr_path: Path, var_name: str) -> None:
    ds = open_zarr_dataset(zarr_path)

    print("\n=== DATASET ===")
    print(ds)

    print("\n=== DIMENSIONS ===")
    for dim, size in ds.sizes.items():
        print(f"{dim}: {size}")

    print("\n=== COORDINATES ===")
    for coord_name in ds.coords:
        coord = ds[coord_name]
        print(
            f"{coord_name}: dims={coord.dims}, "
            f"shape={coord.shape}, dtype={coord.dtype}, "
            f"attrs={dict(coord.attrs)}"
        )

    print("\n=== DATA VARIABLES ===")
    for data_var_name in ds.data_vars:
        data_var = ds[data_var_name]
        chunks = getattr(data_var.data, "chunks", None)
        print(
            f"{data_var_name}: dims={data_var.dims}, "
            f"shape={data_var.shape}, dtype={data_var.dtype}, "
            #f"chunks={chunks}"
        )

    print_attrs("GLOBAL ATTRIBUTES", ds.attrs)

    if var_name not in ds:
        raise KeyError(f"Variable '{var_name}' does not exist in the dataset.")

    da = ds[var_name]

    print("\n=== SELECTED VARIABLE ===")
    print(da)

    print_attrs("VARIABLE ATTRIBUTES", da.attrs)

    print("\n=== VARIABLE ENCODING ===")
    for key, value in da.encoding.items():
        print(f"{key}: {value}")

    print("\n=== INFERRED ISSUE MARKDOWN ===")
    print(build_issue_markdown(ds, da, zarr_path, var_name))

    ds.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a radar Zarr dataset and generate MLCAST issue metadata."
    )
    parser.add_argument("--zarr-path", type=Path, default=DEFAULT_ZARR_PATH)
    parser.add_argument("--var-name", type=str, default=DEFAULT_VAR_NAME)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    inspect_dataset(zarr_path=args.zarr_path, var_name=args.var_name)


if __name__ == "__main__":
    main()