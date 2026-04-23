from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr
import zarr
from pyproj import CRS

from .utils import (
    _open_local_store,
    _rename_only_existing_dims,
    add_crs_metadata,
    build_compressor,
)


DEFAULT_VAR_NAME = "radar"
DEFAULT_STANDARD_NAME: str | None = "equivalent_reflectivity_factor"


def infer_radar_variable(ds: xr.Dataset) -> str:
    """
    Detect the main radar variable:
    - prioritize 3D variables with a time dimension
    - ignore auxiliary variables such as crs
    """
    ignored_vars = {"crs", "spatial_ref", "projection"}

    candidates: list[str] = []

    for var in ds.data_vars:
        if var.lower() in ignored_vars:
            continue

        dims = ds[var].dims
        if "time" in dims and len(dims) == 3:
            candidates.append(var)

    if not candidates:
        non_aux = [v for v in ds.data_vars if v.lower() not in ignored_vars]
        if not non_aux:
            raise ValueError("The dataset does not contain radar data variables.")
        return non_aux[-1]

    preferred_tokens = [
        "dbz",
        "reflect",
        "radar",
        "precip",
        "rain",
        "rate",
        "ppi",
        "acrr",
    ]

    for token in preferred_tokens:
        for var in candidates:
            if token in var.lower():
                return var

    return candidates[0]


def infer_lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    lat_name = None
    lon_name = None

    for name in ds.variables:
        lname = name.lower()
        if lname in {"lat", "latitude"}:
            lat_name = name
        elif lname in {"lon", "longitude"}:
            lon_name = name

    if lat_name is None or lon_name is None:
        raise ValueError("lat/lon variables were not found in the dataset.")

    return lat_name, lon_name


def _rename_spatial_dims(
    da: xr.DataArray,
    lat: xr.DataArray,
    lon: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Force y/x spatial dimension names in the main variable.
    Supports 1D or 2D lat/lon.
    """
    dims = list(da.dims)
    if "time" not in dims:
        raise ValueError(f"The radar variable does not have a time dimension: {dims}")

    spatial_dims = [d for d in dims if d != "time"]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected 2 spatial dimensions, not {len(spatial_dims)}: {dims}")

    rename_map = {}
    if spatial_dims[0] != "y":
        rename_map[spatial_dims[0]] = "y"
    if spatial_dims[1] != "x":
        rename_map[spatial_dims[1]] = "x"

    da = _rename_only_existing_dims(da, rename_map).transpose("time", "y", "x")
    lat = _rename_only_existing_dims(lat, rename_map)
    lon = _rename_only_existing_dims(lon, rename_map)

    return da, lat, lon


def _lat_lon_to_2d(lat: xr.DataArray, lon: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Convert lat/lon to 2D with dims (y, x).
    """
    if lat.dims == ("y",) and lon.dims == ("x",):
        lon2d, lat2d = np.meshgrid(lon.values, lat.values)
        lat2d = xr.DataArray(lat2d, dims=("y", "x"))
        lon2d = xr.DataArray(lon2d, dims=("y", "x"))
        return lat2d, lon2d

    if lat.dims == ("x",) and lon.dims == ("y",):
        lat_vals = np.tile(lat.values[None, :], (len(lon.values), 1))
        lon_vals = np.tile(lon.values[:, None], (1, len(lat.values)))
        lat2d = xr.DataArray(lat_vals, dims=("y", "x"))
        lon2d = xr.DataArray(lon_vals, dims=("y", "x"))
        return lat2d, lon2d

    if set(lat.dims) == {"y", "x"} and set(lon.dims) == {"y", "x"}:
        return lat.transpose("y", "x"), lon.transpose("y", "x")

    raise ValueError(
        f"Unsupported lat/lon format. lat.dims={lat.dims}, lon.dims={lon.dims}"
    )


def standardize_radar_dataset(
    ds: xr.Dataset,
    file_path: str | Path = "",
    var_name: str = DEFAULT_VAR_NAME,
    standard_name: str | None = DEFAULT_STANDARD_NAME,
) -> xr.Dataset:
    """
    Standardize a radar NetCDF into a Dataset with:
      - <var_name>(time, y, x)
      - time
      - y, x
      - lat(y, x), lon(y, x)
    """
    file_path = str(file_path)

    radar_var = infer_radar_variable(ds)
    lat_name, lon_name = infer_lat_lon_names(ds)

    da = ds[radar_var]
    lat = ds[lat_name]
    lon = ds[lon_name]

    da, lat, lon = _rename_spatial_dims(da, lat, lon)
    lat2d, lon2d = _lat_lon_to_2d(lat, lon)

    raw = np.asarray(da.values, dtype="float32").copy()

    fill_candidates = []
    for key in ["_FillValue", "missing_value", "fill_value"]:
        if key in da.attrs and da.attrs[key] is not None:
            fill_candidates.append(float(da.attrs[key]))

    for fv in fill_candidates:
        raw[raw == fv] = np.nan

    scale_factor = da.attrs.get("scale_factor", None)
    add_offset = da.attrs.get("add_offset", None)

    if scale_factor is not None:
        raw = raw * float(scale_factor)
    if add_offset is not None:
        raw = raw + float(add_offset)

    radar_attrs = dict(da.attrs)
    original_standard_name = radar_attrs.get("standard_name", None)

    for key in [
        "_FillValue",
        "missing_value",
        "fill_value",
        "add_offset",
        "scale_factor",
        "standard_name",
    ]:
        radar_attrs.pop(key, None)

    resolved_standard_name = original_standard_name if standard_name is None else standard_name

    out = xr.Dataset(
        data_vars={
            var_name: (("time", "y", "x"), raw),
        },
        coords={
            "time": da["time"].values,
            "y": np.arange(raw.shape[1], dtype=np.int32),
            "x": np.arange(raw.shape[2], dtype=np.int32),
            "lat": (("y", "x"), lat2d.values.astype("float32")),
            "lon": (("y", "x"), lon2d.values.astype("float32")),
        },
        attrs=dict(ds.attrs),
    )

    out[var_name].attrs.update(radar_attrs)

    final_attrs = {
        "long_name": radar_attrs.get("long_name", radar_var),
        "units": radar_attrs.get("units", "unknown"),
        "coordinates": "lat lon",
        "grid_mapping": "spatial_ref",
        "original_variable_name": radar_var,
    }

    if original_standard_name is not None:
        final_attrs["original_standard_name"] = original_standard_name

    if resolved_standard_name is not None:
        final_attrs["standard_name"] = resolved_standard_name

    out[var_name].attrs.update(final_attrs)

    out["lat"].attrs.update(
        {
            "standard_name": "latitude",
            "long_name": lat.attrs.get("long_name", "latitude"),
            "units": lat.attrs.get("units", "degrees_north"),
        }
    )
    out["lon"].attrs.update(
        {
            "standard_name": "longitude",
            "long_name": lon.attrs.get("long_name", "longitude"),
            "units": lon.attrs.get("units", "degrees_east"),
        }
    )

    if "crs" in ds:
        out["spatial_ref"] = xr.DataArray(
            ds["crs"].values,
            attrs=dict(ds["crs"].attrs),
        )

        crs = CRS.from_epsg(4326)

        if "grid_mapping_name" not in out["spatial_ref"].attrs:
            out["spatial_ref"].attrs["grid_mapping_name"] = "latitude_longitude"

        if "crs_wkt" not in out["spatial_ref"].attrs:
            out["spatial_ref"].attrs["crs_wkt"] = crs.to_wkt()

        if "spatial_ref" not in out["spatial_ref"].attrs:
            out["spatial_ref"].attrs["spatial_ref"] = crs.to_wkt()
    else:
        out = add_crs_metadata(out, epsg="EPSG:4326")

    return out


def assert_same_spatial_grid(
    datasets: Iterable[xr.Dataset],
    var_name: str = DEFAULT_VAR_NAME,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> None:
    datasets = list(datasets)
    if not datasets:
        raise ValueError("There are no datasets to compare.")

    ref = datasets[0]
    ref_lat = ref["lat"].values
    ref_lon = ref["lon"].values
    ref_shape = ref[var_name].shape[1:]

    for i, ds in enumerate(datasets[1:], start=1):
        if ds[var_name].shape[1:] != ref_shape:
            raise ValueError(
                f"Inconsistent spatial shape in dataset {i}: "
                f"{ds[var_name].shape[1:]} != {ref_shape}"
            )
        if not np.allclose(ds["lat"].values, ref_lat, rtol=rtol, atol=atol, equal_nan=True):
            raise ValueError(f"Inconsistent lat in dataset {i}")
        if not np.allclose(ds["lon"].values, ref_lon, rtol=rtol, atol=atol, equal_nan=True):
            raise ValueError(f"Inconsistent lon in dataset {i}")


def load_and_standardize_nc_files(
    nc_files: Iterable[str | Path],
    var_name: str = DEFAULT_VAR_NAME,
    standard_name: str | None = DEFAULT_STANDARD_NAME,
    verbose: bool = True,
    inspect_raw_fn=None,
) -> list[xr.Dataset]:
    standardized: list[xr.Dataset] = []

    for fp in nc_files:
        fp = Path(fp)
        if verbose:
            print(f"Reading: {fp}")

        ds = xr.open_dataset(fp, decode_times=True, mask_and_scale=False)

        try:
            radar_var = infer_radar_variable(ds)
            if inspect_raw_fn is not None:
                inspect_raw_fn(ds, radar_var)

            if verbose:
                print(f"Inferred radar variable: {radar_var}")

            std = standardize_radar_dataset(
                ds,
                file_path=fp,
                var_name=var_name,
                standard_name=standard_name,
            )
            standardized.append(std)
        finally:
            ds.close()

    return standardized


def concat_radar_datasets(
    datasets: Iterable[xr.Dataset],
    var_name: str = DEFAULT_VAR_NAME,
) -> xr.Dataset:
    datasets = list(datasets)
    if not datasets:
        raise ValueError("There are no datasets to concatenate.")

    assert_same_spatial_grid(datasets, var_name=var_name)

    var_list = []
    for ds in datasets:
        tmp = ds[[var_name]].reset_coords(names=["lat", "lon"], drop=True)
        var_list.append(tmp)

    ds_all = xr.concat(
        var_list,
        dim="time",
        data_vars="all",
        coords="minimal",
        compat="override",
        combine_attrs="override",
        join="exact",
    ).sortby("time")

    _, idx = np.unique(ds_all["time"].values, return_index=True)
    ds_all = ds_all.isel(time=np.sort(idx))

    ds_all[var_name] = ds_all[var_name].transpose("time", "y", "x")

    ds_all = ds_all.assign_coords(
        lat=(("y", "x"), datasets[0]["lat"].values),
        lon=(("y", "x"), datasets[0]["lon"].values),
    )
    ds_all["lat"].attrs = dict(datasets[0]["lat"].attrs)
    ds_all["lon"].attrs = dict(datasets[0]["lon"].attrs)

    if "spatial_ref" in datasets[0]:
        ds_all["spatial_ref"] = xr.DataArray(
            datasets[0]["spatial_ref"].values,
            attrs=dict(datasets[0]["spatial_ref"].attrs),
        )

    ds_all.attrs = dict(datasets[0].attrs)
    return ds_all


def prepare_radar_dataset_for_zarr(
    ds: xr.Dataset,
    var_name: str = DEFAULT_VAR_NAME,
    epsg: str = "EPSG:4326",
    title: str = "Radar test dataset converted to GeoZarr-like structure",
    institution: str = "AEMET",
    source: str = "AEMET BigData radar product",
    license_name: str = "CC-BY-4.0",
    attribution: str = "Data provided by AEMET",
    mlcast_created_by: str = "Tu Nombre <tu_email@aemet.es>",
    mlcast_created_with: str = "https://github.com/tu-org/tu-repo@v0.1.0",
    mlcast_dataset_version: str = "0.1.0",
    mlcast_dataset_identifier: str = "es-aemet-radar-qa",
    mlcast_dataset_identifier_format: str = "{country_code}-{entity}-{physical_variable}-{time_resolution}-{common_name}",
    time_chunk: int = 1,
) -> xr.Dataset:
    if "spatial_ref" not in ds:
        ds = add_crs_metadata(ds, epsg=epsg)

    ds.attrs.update(
        {
            "title": title,
            "Conventions": "CF-1.10",
            "institution": institution,
            "source": source,
            "history": "Converted from NetCDF radar files to Zarr",
            "license": license_name,
            "attribution": attribution,
            "mlcast_created_on": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "mlcast_created_by": mlcast_created_by,
            "mlcast_created_with": mlcast_created_with,
            "mlcast_dataset_version": mlcast_dataset_version,
            "mlcast_dataset_identifier": mlcast_dataset_identifier,
            "mlcast_dataset_identifier_format": mlcast_dataset_identifier_format,
        }
    )

    ds = ds.chunk({"time": time_chunk, "y": -1, "x": -1})
    return ds


def _clean_variable_attrs_for_zarr(ds: xr.Dataset, var_name: str = DEFAULT_VAR_NAME) -> xr.Dataset:
    ds = ds.copy()

    for var in [var_name, "lat", "lon"]:
        if var in ds:
            for key in ["_FillValue", "missing_value", "fill_value", "add_offset", "scale_factor"]:
                ds[var].attrs.pop(key, None)
                ds[var].encoding.pop(key, None)

    return ds


def build_zarr_v3_compressors(compression_level: int = 5):
    """
    Build Zarr v3 compressors compatible with zarr 3.1.6.
    """
    try:
        from zarr.codecs import ZstdCodec
    except Exception as e:
        raise RuntimeError(
            "Could not import zarr.codecs.ZstdCodec. "
            "Check the zarr 3.x installation."
        ) from e

    return (ZstdCodec(level=compression_level),)


def _save_radar_dataset_to_zarr_v3_sharded(
    ds: xr.Dataset,
    zarr_path: str | Path,
    var_name: str = DEFAULT_VAR_NAME,
    compression_level: int = 5,
    time_chunk: int = 1,
    shard_time: int = 144,
) -> Path:
    """
    Manually write a sharded Zarr v3 store.
    - chunks = logical read unit
    - shards = physical storage/write unit
    """
    zarr_path = Path(zarr_path)
    store = _open_local_store(zarr_path)

    compressors = build_zarr_v3_compressors(compression_level=compression_level)

    radar_chunks = (time_chunk, ds.sizes["y"], ds.sizes["x"])
    radar_shards = (shard_time, ds.sizes["y"], ds.sizes["x"])

    if shard_time < time_chunk:
        raise ValueError("shard_time must be greater than or equal to time_chunk")

    root = zarr.group(store=store, overwrite=True, zarr_format=3)
    root.attrs.update(dict(ds.attrs))

    arr_time = root.create_array(
        "time",
        shape=ds["time"].shape,
        dtype=ds["time"].dtype,
        chunks=ds["time"].shape,
        compressors=compressors,
        dimension_names=("time",),
        overwrite=True,
    )
    arr_y = root.create_array(
        "y",
        shape=ds["y"].shape,
        dtype=ds["y"].dtype,
        chunks=ds["y"].shape,
        compressors=compressors,
        dimension_names=("y",),
        overwrite=True,
    )
    arr_x = root.create_array(
        "x",
        shape=ds["x"].shape,
        dtype=ds["x"].dtype,
        chunks=ds["x"].shape,
        compressors=compressors,
        dimension_names=("x",),
        overwrite=True,
    )
    arr_lat = root.create_array(
        "lat",
        shape=ds["lat"].shape,
        dtype="float32",
        chunks=ds["lat"].shape,
        compressors=compressors,
        dimension_names=("y", "x"),
        overwrite=True,
    )
    arr_lon = root.create_array(
        "lon",
        shape=ds["lon"].shape,
        dtype="float32",
        chunks=ds["lon"].shape,
        compressors=compressors,
        dimension_names=("y", "x"),
        overwrite=True,
    )
    arr_spatial_ref = root.create_array(
        "spatial_ref",
        shape=(),
        dtype=ds["spatial_ref"].dtype,
        chunks=(),
        dimension_names=(),
        overwrite=True,
    )

    arr_time[:] = ds["time"].values
    arr_y[:] = ds["y"].values
    arr_x[:] = ds["x"].values
    arr_lat[:] = ds["lat"].values.astype("float32")
    arr_lon[:] = ds["lon"].values.astype("float32")
    arr_spatial_ref[()] = ds["spatial_ref"].values

    arr_time.attrs.update(dict(ds["time"].attrs))
    arr_y.attrs.update(dict(ds["y"].attrs))
    arr_x.attrs.update(dict(ds["x"].attrs))
    arr_lat.attrs.update(dict(ds["lat"].attrs))
    arr_lon.attrs.update(dict(ds["lon"].attrs))
    arr_spatial_ref.attrs.update(dict(ds["spatial_ref"].attrs))

    arr_radar = root.create_array(
        var_name,
        shape=ds[var_name].shape,
        dtype="float32",
        chunks=radar_chunks,
        shards=radar_shards,
        fill_value=np.nan,
        compressors=compressors,
        dimension_names=("time", "y", "x"),
        overwrite=True,
    )
    arr_radar.attrs.update(dict(ds[var_name].attrs))

    ntime = ds.sizes["time"]
    for t0 in range(0, ntime, shard_time):
        t1 = min(t0 + shard_time, ntime)
        block = ds[var_name].isel(time=slice(t0, t1)).values.astype("float32")
        arr_radar[t0:t1, :, :] = block
        print(f"Written time block {t0}:{t1}")

    print(
        f"Zarr saved to: {zarr_path} "
        f"(format=v3, sharding=True, chunks={radar_chunks}, shards={radar_shards})"
    )
    return zarr_path


def save_radar_dataset_to_zarr(
    ds: xr.Dataset,
    zarr_path: str | Path,
    var_name: str = DEFAULT_VAR_NAME,
    overwrite: bool = True,
    compressor_name: str = "zstd",
    compression_level: int = 5,
    blosc_shuffle: str = "bitshuffle",
    zarr_format: int = 2,
    time_chunk: int = 1,
    use_sharding: bool = False,
    shard_time: int | None = None,
) -> Path:
    zarr_path = Path(zarr_path)

    if overwrite and zarr_path.exists():
        import shutil
        shutil.rmtree(zarr_path)

    if zarr_format == 3 and use_sharding:
        if shard_time is None:
            raise ValueError("You must provide shard_time if use_sharding=True and zarr_format=3")

        return _save_radar_dataset_to_zarr_v3_sharded(
            ds=ds,
            zarr_path=zarr_path,
            var_name=var_name,
            compression_level=compression_level,
            time_chunk=time_chunk,
            shard_time=shard_time,
        )

    encoding = {
        var_name: {
            "dtype": "float32",
            "chunks": (time_chunk, ds.sizes["y"], ds.sizes["x"]),
        },
        "lat": {
            "dtype": "float32",
            "chunks": (ds.sizes["y"], ds.sizes["x"]),
        },
        "lon": {
            "dtype": "float32",
            "chunks": (ds.sizes["y"], ds.sizes["x"]),
        },
    }

    if zarr_format == 2:
        compressor = build_compressor(
            compressor_name=compressor_name,
            compression_level=compression_level,
            blosc_shuffle=blosc_shuffle,
        )

        encoding[var_name]["compressor"] = compressor
        encoding[var_name]["_FillValue"] = np.nan
        encoding["lat"]["compressor"] = compressor
        encoding["lon"]["compressor"] = compressor

        ds.to_zarr(
            zarr_path,
            mode="w",
            consolidated=True,
            encoding=encoding,
            zarr_format=2,
        )

    elif zarr_format == 3:
        encoding[var_name]["fill_value"] = np.nan

        ds.to_zarr(
            zarr_path,
            mode="w",
            consolidated=False,
            encoding=encoding,
            zarr_format=3,
        )

    else:
        raise ValueError("zarr_format must be 2 or 3")

    print(
        f"Zarr saved to: {zarr_path} "
        f"(format=v{zarr_format}, sharding={use_sharding})"
    )
    return zarr_path


def build_radar_zarr_from_nc_files(
    nc_files: Iterable[str | Path],
    zarr_path: str | Path,
    var_name: str = DEFAULT_VAR_NAME,
    standard_name: str | None = DEFAULT_STANDARD_NAME,
    epsg: str = "EPSG:4326",
    overwrite: bool = True,
    verbose: bool = True,
    inspect: bool = True,
    title: str = "Radar test dataset converted to GeoZarr-like structure",
    institution: str = "AEMET",
    source: str = "AEMET BigData radar product",
    license_name: str = "CC-BY-4.0",
    attribution: str = "Data provided by AEMET",
    mlcast_created_by: str = "Tu Nombre <tu_email@aemet.es>",
    mlcast_created_with: str = "https://github.com/tu-org/tu-repo@v0.1.0",
    mlcast_dataset_version: str = "0.1.0",
    mlcast_dataset_identifier: str = "es-aemet-radar-qa",
    mlcast_dataset_identifier_format: str = "{country_code}-{entity}-{physical_variable}-{time_resolution}-{common_name}",
    compressor_name: str = "zstd",
    compression_level: int = 5,
    blosc_shuffle: str = "bitshuffle",
    zarr_format: int = 2,
    time_chunk: int = 1,
    use_sharding: bool = False,
    shard_time: int | None = None,
    inspect_raw_fn=None,
    inspect_state_fn=None,
) -> xr.Dataset:
    standardized = load_and_standardize_nc_files(
        nc_files,
        var_name=var_name,
        standard_name=standard_name,
        verbose=verbose,
        inspect_raw_fn=inspect_raw_fn,
    )

    if inspect and inspect_state_fn is not None:
        print(f"\nNumber of standardized datasets: {len(standardized)}")
        for i, ds in enumerate(standardized[:2]):
            inspect_state_fn(ds, label=f"standardized[{i}]", var_name=var_name, time_index=0)

    ds_all = concat_radar_datasets(standardized, var_name=var_name)

    if inspect and inspect_state_fn is not None:
        inspect_state_fn(
            ds_all,
            label="after concat_radar_datasets",
            var_name=var_name,
            time_index=0,
        )

    ds_all = prepare_radar_dataset_for_zarr(
        ds_all,
        var_name=var_name,
        epsg=epsg,
        title=title,
        institution=institution,
        source=source,
        license_name=license_name,
        attribution=attribution,
        mlcast_created_by=mlcast_created_by,
        mlcast_created_with=mlcast_created_with,
        mlcast_dataset_version=mlcast_dataset_version,
        mlcast_dataset_identifier=mlcast_dataset_identifier,
        mlcast_dataset_identifier_format=mlcast_dataset_identifier_format,
        time_chunk=time_chunk,
    )

    ds_all = _clean_variable_attrs_for_zarr(ds_all, var_name=var_name)

    if inspect and inspect_state_fn is not None:
        inspect_state_fn(
            ds_all,
            label="after prepare_radar_dataset_for_zarr",
            var_name=var_name,
            time_index=0,
        )

    save_radar_dataset_to_zarr(
        ds_all,
        zarr_path=zarr_path,
        var_name=var_name,
        overwrite=overwrite,
        compressor_name=compressor_name,
        compression_level=compression_level,
        blosc_shuffle=blosc_shuffle,
        zarr_format=zarr_format,
        time_chunk=time_chunk,
        use_sharding=use_sharding,
        shard_time=shard_time,
    )

    if inspect and inspect_state_fn is not None:
        consolidated = True if zarr_format == 2 else False

        inspect_state_fn(
            zarr_path,
            label="after save_radar_dataset_to_zarr (reopen decode_cf=False)",
            var_name=var_name,
            time_index=0,
            decode_cf_for_zarr=False,
            consolidated=consolidated,
        )
        inspect_state_fn(
            zarr_path,
            label="after save_radar_dataset_to_zarr (reopen decode_cf=True)",
            var_name=var_name,
            time_index=0,
            decode_cf_for_zarr=True,
            consolidated=consolidated,
        )

    return ds_all