from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from dask.diagnostics import ProgressBar
from zarr.codecs import ZstdCodec

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# =============================================================================
# CONFIG
# =============================================================================

def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError("The YAML configuration must contain a dictionary.")

    return cfg


def detect_zarr_format(path: Path) -> int:
    return 3 if (path / "zarr.json").exists() else 2


def open_zarr_dataset(path: Path) -> xr.Dataset:
    zarr_format = detect_zarr_format(path)
    return xr.open_zarr(
        path,
        consolidated=False if zarr_format == 3 else True,
        zarr_format=zarr_format,
        chunks={"time": 1},
    )


# =============================================================================
# TIME / PRODUCTS
# =============================================================================

def infer_time_step_hours(ds: xr.Dataset, fallback_minutes: float = 10.0) -> float:
    """Infer a strictly positive temporal resolution in hours.

    Radar composites may contain duplicated timestamps or irregular gaps. For the
    rolling-window calculations we need the nominal radar timestep, not a zero
    difference caused by duplicates. If inference fails, the configured fallback
    is used.
    """
    fallback_hours = float(fallback_minutes) / 60.0

    if fallback_hours <= 0:
        raise ValueError(
            f"time_step_minutes must be positive, got {fallback_minutes!r}."
        )

    if "time" not in ds.coords or ds.sizes.get("time", 0) < 2:
        print(
            f"[TIME] Could not infer timestep from data. Using fallback: "
            f"{fallback_minutes} minutes."
        )
        return fallback_hours

    time_values = pd.DatetimeIndex(ds["time"].values).sort_values()

    # Drop duplicated timestamps before calculating differences.
    time_values = time_values.drop_duplicates()

    if len(time_values) < 2:
        print(
            f"[TIME] Less than two unique timestamps. Using fallback: "
            f"{fallback_minutes} minutes."
        )
        return fallback_hours

    diffs_seconds = np.diff(time_values.view("int64")) / 1.0e9
    diffs_seconds = diffs_seconds[np.isfinite(diffs_seconds)]
    diffs_seconds = diffs_seconds[diffs_seconds > 0]

    if len(diffs_seconds) == 0:
        print(
            f"[TIME] No positive time differences found. Using fallback: "
            f"{fallback_minutes} minutes."
        )
        return fallback_hours

    inferred_hours = float(np.nanmedian(diffs_seconds) / 3600.0)

    if not np.isfinite(inferred_hours) or inferred_hours <= 0:
        print(
            f"[TIME] Invalid inferred timestep {inferred_hours!r}. Using fallback: "
            f"{fallback_minutes} minutes."
        )
        return fallback_hours

    print(f"[TIME] Inferred timestep: {inferred_hours * 60.0:.3f} minutes.")
    return inferred_hours


def rolling_window_size(hours: float, time_step_hours: float) -> int:
    if not np.isfinite(time_step_hours) or time_step_hours <= 0:
        raise ValueError(
            f"time_step_hours must be positive, got {time_step_hours!r}. "
            "Set a positive time_step_minutes value in the YAML config."
        )

    return max(1, int(round(hours / time_step_hours)))


def compute_event_products(ds: xr.Dataset, cfg: dict[str, Any]) -> xr.Dataset:
    rainfall_var = cfg.get("rainfall_rate_var", "rainfall_rate")
    reflectivity_var = cfg.get("reflectivity_var", "reflectivity_dbz")
    fallback_minutes = float(cfg.get("time_step_minutes", 10.0))

    if rainfall_var not in ds:
        raise KeyError(
            f"Rainfall-rate variable {rainfall_var!r} was not found. "
            f"Available variables: {list(ds.data_vars)}"
        )

    time_step_hours = infer_time_step_hours(ds, fallback_minutes=fallback_minutes)

    rainfall_rate = ds[rainfall_var].where(ds[rainfall_var] > 0)
    rain_step = rainfall_rate * time_step_hours

    acc_total = rain_step.sum("time", skipna=True)

    products: dict[str, xr.DataArray] = {
        "acc_total_mm": acc_total.astype("float32"),
    }

    for hours in cfg.get("accumulation_windows_hours", [1, 3, 6, 12, 24]):
        window = rolling_window_size(float(hours), time_step_hours)
        name = f"max_{int(hours)}h_mm"
        products[name] = (
            rain_step.rolling(time=window, min_periods=1)
            .sum()
            .max("time", skipna=True)
            .astype("float32")
        )
        products[name].attrs = {
            "long_name": f"Maximum moving accumulated rainfall over {hours} hours",
            "units": "mm",
            "window_hours": float(hours),
            "window_steps": int(window),
        }

    for threshold in cfg.get("rainfall_rate_persistence_thresholds_mm_h", [5, 10, 20]):
        threshold_float = float(threshold)
        safe_name = str(threshold).replace(".", "p")
        name = f"persistence_rainfall_rate_ge_{safe_name}mmh_h"
        products[name] = (
            (rainfall_rate >= threshold_float).sum("time") * time_step_hours
        ).astype("float32")
        products[name].attrs = {
            "long_name": f"Persistence of rainfall rate >= {threshold_float} mm h-1",
            "units": "h",
            "threshold_mm_h": threshold_float,
        }

    if reflectivity_var in ds:
        reflectivity = ds[reflectivity_var]
        products["max_reflectivity_dbz"] = reflectivity.max("time", skipna=True).astype("float32")
        products["max_reflectivity_dbz"].attrs = {
            "long_name": "Maximum radar reflectivity during the event",
            "units": "dBZ",
        }

        for threshold in cfg.get("reflectivity_persistence_thresholds_dbz", [35, 40, 45]):
            threshold_float = float(threshold)
            safe_name = str(threshold).replace(".", "p")
            name = f"persistence_reflectivity_ge_{safe_name}dbz_h"
            products[name] = (
                (reflectivity >= threshold_float).sum("time") * time_step_hours
            ).astype("float32")
            products[name].attrs = {
                "long_name": f"Persistence of reflectivity >= {threshold_float} dBZ",
                "units": "h",
                "threshold_dbz": threshold_float,
            }

    if "radar_count_available" in ds:
        products["max_radar_count_available"] = (
            ds["radar_count_available"].max("time", skipna=True).astype("int16")
        )
        products["max_radar_count_available"].attrs = {
            "long_name": "Maximum number of radars with valid data",
            "units": "1",
        }

    ds_products = xr.Dataset(products)

    # Re-attach spatial coordinates explicitly.
    # Some xarray reductions may leave x/y as dimensions but not as coordinates,
    # especially after writing/reopening intermediate Zarr stores or when coordinates
    # are stored as data variables in the source dataset.
    for coord in ["x", "y", "lat", "lon"]:
        if coord in ds.coords:
            ds_products = ds_products.assign_coords({coord: ds[coord]})
        elif coord in ds.data_vars:
            ds_products = ds_products.assign_coords({coord: ds[coord]})

    if "spatial_ref" in ds:
        ds_products["spatial_ref"] = ds["spatial_ref"]
    elif "spatial_ref" in ds.data_vars:
        ds_products["spatial_ref"] = ds["spatial_ref"]

    ds_products.attrs.update(dict(ds.attrs))
    ds_products.attrs["analysis_time_step_hours"] = time_step_hours
    ds_products.attrs["analysis_note"] = (
        "Rainfall accumulations are computed from rainfall_rate in mm h-1. "
        "Reflectivity-derived products are computed from reflectivity_dbz when available."
    )

    ds_products["acc_total_mm"].attrs = {
        "long_name": "Total radar-estimated accumulated rainfall during the event",
        "units": "mm",
    }

    return ds_products


def build_impact_classification(products: xr.Dataset, cfg: dict[str, Any]) -> xr.DataArray:
    impact_cfg = cfg.get("impact_classification", {})

    acc_medium = float(impact_cfg.get("acc_total_medium_mm", 50.0))
    acc_high = float(impact_cfg.get("acc_total_high_mm", 100.0))
    max1h_medium = float(impact_cfg.get("max_1h_medium_mm", 20.0))
    max1h_high = float(impact_cfg.get("max_1h_high_mm", 40.0))

    if "max_1h_mm" not in products:
        raise KeyError("max_1h_mm is required to build impact classification.")

    impact = xr.where(
        (products["acc_total_mm"] >= acc_high) | (products["max_1h_mm"] >= max1h_high),
        3,
        xr.where(
            (products["acc_total_mm"] >= acc_medium)
            | (products["max_1h_mm"] >= max1h_medium),
            2,
            xr.where(products["acc_total_mm"] > 0, 1, 0),
        ),
    ).astype("int16")

    impact.name = "meteorological_impact_class"
    impact.attrs = {
        "long_name": "Auxiliary meteorological impact class",
        "description": (
            "Auxiliary classification based only on radar-estimated rainfall. "
            "It must not be interpreted as an automatic damage assessment."
        ),
        "flag_values": "0, 1, 2, 3",
        "flag_meanings": "no_relevant_signal low medium high",
        "acc_total_medium_mm": acc_medium,
        "acc_total_high_mm": acc_high,
        "max_1h_medium_mm": max1h_medium,
        "max_1h_high_mm": max1h_high,
    }

    return impact


# =============================================================================
# PLOTTING
# =============================================================================

def get_data_crs(ds: xr.Dataset, epsg: str | None = None):
    if epsg is not None:
        epsg_clean = str(epsg).upper().replace("EPSG:", "")
        if epsg_clean == "4326":
            return ccrs.PlateCarree()
        return ccrs.epsg(int(epsg_clean))

    if "spatial_ref" in ds:
        epsg_code = ds["spatial_ref"].attrs.get("epsg_code")
        if epsg_code is not None:
            epsg_clean = str(epsg_code).upper().replace("EPSG:", "")
            if epsg_clean == "4326":
                return ccrs.PlateCarree()
            return ccrs.epsg(int(epsg_clean))

    return ccrs.epsg(25830)


def ensure_plot_coordinates(da: xr.DataArray, products: xr.Dataset) -> xr.DataArray:
    """Ensure a spatial field has x/y coordinates attached before plotting."""
    out = da

    for coord in ["x", "y", "lat", "lon"]:
        if coord not in out.coords and coord in products.coords:
            out = out.assign_coords({coord: products[coord]})
        elif coord not in out.coords and coord in products.data_vars:
            out = out.assign_coords({coord: products[coord]})

    return out


def plot_spatial_field(
    da: xr.DataArray,
    output_path: Path,
    title: str,
    label: str,
    epsg: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_crs = get_data_crs(da.to_dataset(name="tmp"), epsg=epsg)

    fig = plt.figure(figsize=(9, 8))
    ax = plt.axes(projection=data_crs)

    if "x" not in da.coords or "y" not in da.coords:
        raise KeyError("The DataArray must contain x and y coordinates for plotting.")

    x = da["x"].values
    y = da["y"].values

    dx = float(np.nanmedian(np.diff(x))) if len(x) > 1 else 0.0
    dy = float(np.nanmedian(np.diff(y))) if len(y) > 1 else 0.0

    extent = [
        float(np.nanmin(x) - dx / 2),
        float(np.nanmax(x) + dx / 2),
        float(np.nanmin(y) - dy / 2),
        float(np.nanmax(y) + dy / 2),
    ]

    mesh = ax.imshow(
        da.values,
        origin="lower",
        extent=extent,
        transform=data_crs,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_extent(extent, crs=data_crs)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.6)

    admin1 = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale="10m",
        facecolor="none",
    )
    ax.add_feature(admin1, edgecolor="black", linewidth=0.5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(title)
    cbar = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.06, fraction=0.05)
    cbar.set_label(label)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_maps(products: xr.Dataset, cfg: dict[str, Any]) -> None:
    maps_dir = Path(cfg["output_dir"]) / "maps"
    epsg = str(cfg.get("epsg", "25830"))

    map_specs = cfg.get("maps", {})

    default_specs = {
        "acc_total_mm": ("Total radar-estimated accumulated rainfall", "mm"),
        "max_1h_mm": ("Maximum 1-hour radar-estimated rainfall", "mm / 1 h"),
        "max_3h_mm": ("Maximum 3-hour radar-estimated rainfall", "mm / 3 h"),
        "max_6h_mm": ("Maximum 6-hour radar-estimated rainfall", "mm / 6 h"),
        "max_reflectivity_dbz": ("Maximum radar reflectivity", "dBZ"),
        "max_radar_count_available": ("Maximum number of available radars", "number of radars"),
        "meteorological_impact_class": ("Auxiliary meteorological impact class", "class"),
    }

    for name, (title, label) in default_specs.items():
        if name not in products:
            continue

        spec = map_specs.get(name, {}) if isinstance(map_specs, dict) else {}
        da_to_plot = ensure_plot_coordinates(products[name], products)
        plot_spatial_field(
            da_to_plot,
            output_path=maps_dir / f"{name}.png",
            title=spec.get("title", title),
            label=spec.get("label", label),
            epsg=epsg,
            vmin=spec.get("vmin"),
            vmax=spec.get("vmax"),
        )

    for name in products.data_vars:
        if name.startswith("persistence_rainfall_rate") or name.startswith("persistence_reflectivity"):
            units = products[name].attrs.get("units", "h")
            long_name = products[name].attrs.get("long_name", name)
            da_to_plot = ensure_plot_coordinates(products[name], products)
            plot_spatial_field(
                da_to_plot,
                output_path=maps_dir / f"{name}.png",
                title=long_name,
                label=units,
                epsg=epsg,
            )


# =============================================================================
# SUMMARY TABLES
# =============================================================================

def summarize_products(products: xr.Dataset, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []

    for name, da in products.data_vars.items():
        if not {"y", "x"}.issubset(da.dims):
            continue

        values = da.values
        rows.append(
            {
                "variable": name,
                "min": float(np.nanmin(values)) if np.isfinite(values).any() else np.nan,
                "p50": float(np.nanpercentile(values, 50)) if np.isfinite(values).any() else np.nan,
                "p90": float(np.nanpercentile(values, 90)) if np.isfinite(values).any() else np.nan,
                "p95": float(np.nanpercentile(values, 95)) if np.isfinite(values).any() else np.nan,
                "p99": float(np.nanpercentile(values, 99)) if np.isfinite(values).any() else np.nan,
                "max": float(np.nanmax(values)) if np.isfinite(values).any() else np.nan,
                "units": da.attrs.get("units", ""),
                "long_name": da.attrs.get("long_name", ""),
            }
        )

    return pd.DataFrame(rows)


def write_summary_tables(products: xr.Dataset, cfg: dict[str, Any]) -> None:
    tables_dir = Path(cfg["output_dir"]) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_products(products, cfg)
    summary.to_csv(tables_dir / "summary_spatial_products.csv", index=False)


# =============================================================================
# ZARR OUTPUT
# =============================================================================

def build_encoding(ds: xr.Dataset, cfg: dict[str, Any]) -> dict[str, Any]:
    compressor = ZstdCodec(level=int(cfg.get("compression_level", 5)))
    zarr_format = int(cfg.get("zarr_version", 3))

    y_size = ds.sizes["y"]
    x_size = ds.sizes["x"]

    encoding: dict[str, Any] = {}

    for name, da in ds.data_vars.items():
        if da.dims == ("y", "x"):
            item: dict[str, Any] = {
                "chunks": (y_size, x_size),
                "dtype": da.dtype,
            }
            if zarr_format == 3:
                item["compressors"] = (compressor,)
            if np.issubdtype(da.dtype, np.floating):
                item["_FillValue"] = np.float32(np.nan)
            encoding[name] = item

    for coord in ["lat", "lon"]:
        if coord in ds.coords and ds[coord].dims == ("y", "x"):
            item = {
                "chunks": (y_size, x_size),
                "dtype": np.float32,
                "_FillValue": np.float32(np.nan),
            }
            if zarr_format == 3:
                item["compressors"] = (compressor,)
            encoding[coord] = item

    return encoding


def write_products_zarr(products: xr.Dataset, cfg: dict[str, Any]) -> None:
    output_zarr = cfg.get("products_zarr_out")
    if not output_zarr:
        return

    output_zarr_path = Path(output_zarr)
    if output_zarr_path.exists():
        if bool(cfg.get("overwrite_products", True)):
            if output_zarr_path.is_dir():
                import shutil
                shutil.rmtree(output_zarr_path)
            else:
                output_zarr_path.unlink()
        else:
            raise FileExistsError(f"Products Zarr already exists: {output_zarr_path}")

    output_zarr_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = build_encoding(products, cfg)

    print(f"[WRITE] Products Zarr: {output_zarr_path}")
    with ProgressBar():
        products.to_zarr(
            output_zarr_path,
            mode="w",
            consolidated=False,
            zarr_format=int(cfg.get("zarr_version", 3)),
            encoding=encoding,
            compute=True,
            align_chunks=True,
            safe_chunks=False,
        )


# =============================================================================
# MAIN
# =============================================================================

def run_analysis(cfg: dict[str, Any]) -> None:
    composite_zarr = Path(cfg["composite_zarr"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[OPEN] Composite Zarr: {composite_zarr}")
    ds = open_zarr_dataset(composite_zarr)

    try:
        products = compute_event_products(ds, cfg)
        products["meteorological_impact_class"] = build_impact_classification(products, cfg)

        write_products_zarr(products, cfg)
        write_maps(products, cfg)
        write_summary_tables(products, cfg)

        print(f"[OK] Analysis outputs written to: {output_dir}")

    finally:
        ds.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a radar event composite and generate report-ready products."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    run_analysis(cfg)


if __name__ == "__main__":
    main()
