from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .radar_processing import DEFAULT_VAR_NAME


def open_radar_zarr(
    zarr_path: str | Path,
    consolidated: bool | None = None,
) -> xr.Dataset:
    return xr.open_zarr(zarr_path, consolidated=consolidated)


def quick_validate_radar_zarr(
    zarr_path: str | Path,
    var_name: str = DEFAULT_VAR_NAME,
    consolidated: bool | None = None,
) -> None:
    print("\n--- QUICK VALIDATION ---")
    ds = open_radar_zarr(zarr_path, consolidated=consolidated)

    print(ds)

    if var_name not in ds:
        raise ValueError(f"Missing variable '{var_name}'")

    if ds[var_name].dims != ("time", "y", "x"):
        raise ValueError(f"Incorrect dims in {var_name}: {ds[var_name].dims}")

    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError("Missing lat/lon coordinates")

    if ds["lat"].dims != ("y", "x"):
        raise ValueError(f"Incorrect lat dims: {ds['lat'].dims}")
    if ds["lon"].dims != ("y", "x"):
        raise ValueError(f"Incorrect lon dims: {ds['lon'].dims}")

    if "spatial_ref" not in ds:
        raise ValueError("Missing spatial_ref variable")

    wkt = ds["spatial_ref"].attrs.get("crs_wkt") or ds["spatial_ref"].attrs.get("spatial_ref")
    if wkt is None:
        raise ValueError("Missing crs_wkt/spatial_ref in spatial_ref")

    from pyproj import CRS
    import cartopy.crs as ccrs

    crs = CRS.from_wkt(wkt)
    print("CRS read from WKT:", crs)

    cartopy_crs = ccrs.PlateCarree()
    print("Cartopy CRS used for visualization:", cartopy_crs)

    chunks = ds[var_name].encoding.get("chunks")
    print(f"{var_name} chunks:", chunks)

    nan_count = int(np.isnan(ds[var_name].values).sum())
    print("Number of NaNs:", nan_count)

    print("Quick validation OK")


def inspect_raw_radar_dataset(ds: xr.Dataset, radar_var: str) -> None:
    da = ds[radar_var]

    print("\n--- RAW NETCDF INSPECTION ---")
    print("variable:", radar_var)
    print("dims:", da.dims)
    print("shape:", da.shape)
    print("dtype:", da.dtype)
    print("attrs:", dict(da.attrs))
    print("encoding:", dict(da.encoding))

    vals = np.asarray(da.values)

    print("raw min:", np.nanmin(vals))
    print("raw max:", np.nanmax(vals))
    print("raw unique sample:", np.unique(vals)[:20])

    finite = np.isfinite(vals)
    print("raw finite:", int(finite.sum()), "out of", vals.size)


def inspect_radar_dataset_in_memory(
    ds: xr.Dataset,
    var_name: str = DEFAULT_VAR_NAME,
    time_index: int = 0,
) -> None:
    field = ds[var_name].isel(time=time_index).compute().values
    finite = np.isfinite(field)

    print("\n--- IN-MEMORY INSPECTION ---")
    print("shape:", field.shape)
    print("dtype:", field.dtype)
    print("n_total:", field.size)
    print("n_finite:", int(finite.sum()))
    print("n_nan:", int(np.isnan(field).sum()))

    if finite.any():
        vals = field[finite]
        print("min:", float(vals.min()))
        print("max:", float(vals.max()))
        print("p01:", float(np.percentile(vals, 1)))
        print("p50:", float(np.percentile(vals, 50)))
        print("p99:", float(np.percentile(vals, 99)))


def inspect_radar_state(
    obj: xr.Dataset | str | Path,
    label: str,
    var_name: str = DEFAULT_VAR_NAME,
    time_index: int = 0,
    decode_cf_for_zarr: bool = True,
    consolidated: bool | None = None,
) -> None:
    print(f"\n{'=' * 80}")
    print(f"INSPECTION: {label}")
    print(f"{'=' * 80}")

    if isinstance(obj, xr.Dataset):
        ds = obj
        source = "in-memory dataset"
    else:
        ds = xr.open_zarr(obj, consolidated=consolidated, decode_cf=decode_cf_for_zarr)
        source = (
            f"zarr (decode_cf={decode_cf_for_zarr}, consolidated={consolidated})"
        )

    print("Source:", source)
    print(ds)

    if var_name not in ds:
        print(f"Variable '{var_name}' does not exist")
        return

    field = ds[var_name].isel(time=time_index).load().values
    finite = np.isfinite(field)

    print("\n--- FIELD STATISTICS ---")
    print("shape:", field.shape)
    print("dtype:", field.dtype)
    print("n_total:", field.size)
    print("n_finite:", int(finite.sum()))
    print("n_nan:", int(np.isnan(field).sum()))

    print("\n--- VARIABLE ATTRS ---")
    print(ds[var_name].attrs)

    print("\n--- VARIABLE ENCODING ---")
    print(ds[var_name].encoding)

    if finite.any():
        vals = field[finite]
        print("\n--- FINITE VALUE RANGE ---")
        print("min:", float(vals.min()))
        print("max:", float(vals.max()))
        print("mean:", float(vals.mean()))
        print("p01:", float(np.percentile(vals, 1)))
        print("p05:", float(np.percentile(vals, 5)))
        print("p50:", float(np.percentile(vals, 50)))
        print("p95:", float(np.percentile(vals, 95)))
        print("p99:", float(np.percentile(vals, 99)))
        print("unique sample:", np.unique(vals)[:20])
    else:
        print("\nThere are no finite values in this state.")


def inspect_radar_zarr_raw(
    zarr_path: str | Path,
    var_name: str = DEFAULT_VAR_NAME,
    time_index: int = 0,
    consolidated: bool | None = None,
) -> None:
    ds = xr.open_zarr(zarr_path, consolidated=consolidated, decode_cf=False)
    field = ds[var_name].isel(time=time_index).load().values
    finite = np.isfinite(field)

    print("\n--- RAW ZARR INSPECTION (decode_cf=False) ---")
    print("shape:", field.shape)
    print("dtype:", field.dtype)
    print("n_total:", field.size)
    print("n_finite:", int(finite.sum()))
    print("n_nan:", int(np.isnan(field).sum()))

    if finite.any():
        vals = field[finite]
        print("min:", float(vals.min()))
        print("max:", float(vals.max()))
        print("p01:", float(np.percentile(vals, 1)))
        print("p50:", float(np.percentile(vals, 50)))
        print("p99:", float(np.percentile(vals, 99)))


def inspect_radar_zarr(
    zarr_path: str | Path,
    var_name: str = DEFAULT_VAR_NAME,
    time_index: int = 0,
    consolidated: bool | None = None,
) -> None:
    ds = open_radar_zarr(zarr_path, consolidated=consolidated)
    field = ds[var_name].isel(time=time_index).load().values

    finite = np.isfinite(field)
    n_total = field.size
    n_finite = int(finite.sum())
    n_nan = int(np.isnan(field).sum())

    print("\n--- FIELD INSPECTION ---")
    print("shape:", field.shape)
    print("dtype:", field.dtype)
    print("n_total:", n_total)
    print("n_finite:", n_finite)
    print("n_nan:", n_nan)

    if n_finite > 0:
        vals = field[finite]
        print("min:", float(vals.min()))
        print("max:", float(vals.max()))
        print("mean:", float(vals.mean()))
        print("p01:", float(np.percentile(vals, 1)))
        print("p05:", float(np.percentile(vals, 5)))
        print("p50:", float(np.percentile(vals, 50)))
        print("p95:", float(np.percentile(vals, 95)))
        print("p99:", float(np.percentile(vals, 99)))
    else:
        print("There are no finite values in the field.")


def plot_radar_from_zarr(
    zarr_path: str | Path,
    var_name: str = DEFAULT_VAR_NAME,
    time_index: int = 0,
    output_png: str | Path | None = None,
    figsize: tuple[int, int] = (8, 6),
    cmap: str = "viridis",
    add_colorbar: bool = True,
    consolidated: bool | None = None,
) -> None:
    ds = open_radar_zarr(zarr_path, consolidated=consolidated)

    field = ds[var_name].isel(time=time_index)
    lon = ds["lon"]
    lat = ds["lat"]
    time_value = ds["time"].isel(time=time_index).values

    fig, ax = plt.subplots(figsize=figsize)

    mesh = ax.pcolormesh(
        lon.values,
        lat.values,
        field.values,
        shading="auto",
        cmap=cmap,
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{var_name} at time index {time_index}\n{time_value}")

    if add_colorbar:
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label(ds[var_name].attrs.get("units", ""))

    plt.tight_layout()

    if output_png is not None:
        output_png = Path(output_png)
        fig.savefig(output_png, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {output_png}")

    plt.show()


def plot_radar_from_zarr_cartopy(
    zarr_path: str | Path,
    var_name: str = DEFAULT_VAR_NAME,
    time_index: int = 0,
    output_png: str | Path | None = None,
    figsize: tuple[int, int] = (9, 7),
    cmap: str = "viridis",
    add_coastlines: bool = True,
    consolidated: bool | None = None,
) -> None:
    import cartopy.crs as ccrs

    ds = open_radar_zarr(zarr_path, consolidated=consolidated)

    field = ds[var_name].isel(time=time_index)
    lon = ds["lon"]
    lat = ds["lat"]
    time_value = ds["time"].isel(time=time_index).values

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})

    mesh = ax.pcolormesh(
        lon.values,
        lat.values,
        field.values,
        transform=proj,
        shading="auto",
        cmap=cmap,
    )

    if add_coastlines:
        ax.coastlines(resolution="10m")

    ax.set_title(f"{var_name} at time index {time_index}\n{time_value}")
    ax.gridlines(draw_labels=True)

    cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.85)
    cbar.set_label(ds[var_name].attrs.get("units", ""))

    plt.tight_layout()

    if output_png is not None:
        output_png = Path(output_png)
        fig.savefig(output_png, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {output_png}")

    plt.show()