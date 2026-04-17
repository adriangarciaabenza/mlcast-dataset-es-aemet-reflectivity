from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numcodecs import Blosc, Zstd
from pyproj import CRS


DEFAULT_VAR_NAME = "radar"
DEFAULT_STANDARD_NAME: str | None = "equivalent_reflectivity_factor"


def clean_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    return path


def find_nc_files(path: str | Path) -> list[Path]:
    path = Path(path)
    return sorted(path.rglob("*.nc"))


def infer_radar_variable(ds: xr.Dataset) -> str:
    """
    Detecta la variable principal radar:
    - prioriza variables 3D con dimensión time
    - ignora variables auxiliares como crs
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
            raise ValueError("El dataset no contiene variables de datos radar.")
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
        raise ValueError("No se han encontrado variables lat/lon en el dataset.")

    return lat_name, lon_name


def _rename_only_existing_dims(da: xr.DataArray, rename_map: dict[str, str]) -> xr.DataArray:
    """
    Renombra solo las dimensiones que realmente existen en la DataArray.
    """
    local_map = {d: rename_map[d] for d in da.dims if d in rename_map}
    if local_map:
        da = da.rename(local_map)
    return da


def _rename_spatial_dims(
    da: xr.DataArray,
    lat: xr.DataArray,
    lon: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Fuerza nombres espaciales y/x en la variable principal.
    Soporta lat/lon 1D o 2D.
    """
    dims = list(da.dims)
    if "time" not in dims:
        raise ValueError(f"La variable radar no tiene dimensión time: {dims}")

    spatial_dims = [d for d in dims if d != "time"]
    if len(spatial_dims) != 2:
        raise ValueError(f"Se esperaban 2 dimensiones espaciales, no {len(spatial_dims)}: {dims}")

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
    Convierte lat/lon a 2D con dims (y, x).

    Casos soportados:
    - lat(y), lon(x)
    - lat(x), lon(y)  [caso raro]
    - lat(y,x), lon(y,x)
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
        f"Formato de lat/lon no soportado. lat.dims={lat.dims}, lon.dims={lon.dims}"
    )


def _copy_attrs_safe(src: xr.DataArray, dst: xr.DataArray) -> xr.DataArray:
    dst.attrs.update(dict(src.attrs))
    return dst


def build_compressor(
    compressor_name: str = "zstd",
    compression_level: int = 5,
    blosc_shuffle: str = "bitshuffle",
):
    """
    Construye un compresor numcodecs configurable.

    Parámetros
    ----------
    compressor_name : {"zstd", "blosc"}
        - "zstd"  -> usa numcodecs.Zstd directamente
        - "blosc" -> usa numcodecs.Blosc con cname="zstd"
    compression_level : int
        Nivel de compresión.
    blosc_shuffle : {"bitshuffle", "shuffle", "noshuffle"}
        Solo aplica cuando compressor_name="blosc".
    """
    compressor_name = compressor_name.lower().strip()

    if compressor_name == "zstd":
        return Zstd(level=compression_level)

    if compressor_name == "blosc":
        shuffle_map = {
            "bitshuffle": Blosc.BITSHUFFLE,
            "shuffle": Blosc.SHUFFLE,
            "noshuffle": Blosc.NOSHUFFLE,
        }

        shuffle_value = shuffle_map.get(blosc_shuffle.lower())
        if shuffle_value is None:
            raise ValueError(
                f"blosc_shuffle no válido: {blosc_shuffle}. "
                "Usa 'bitshuffle', 'shuffle' o 'noshuffle'."
            )

        return Blosc(
            cname="zstd",
            clevel=compression_level,
            shuffle=shuffle_value,
        )

    raise ValueError(
        f"compressor_name no válido: {compressor_name}. "
        "Usa 'zstd' o 'blosc'."
    )


def standardize_radar_dataset(
    ds: xr.Dataset,
    file_path: str | Path = "",
    var_name: str = DEFAULT_VAR_NAME,
    standard_name: str | None = DEFAULT_STANDARD_NAME,
) -> xr.Dataset:
    """
    Estandariza un NetCDF radar a un Dataset con:
      - <var_name>(time, y, x)
      - time
      - y, x
      - lat(y, x), lon(y, x)

    Soporta:
      - lat/lon 1D
      - lat/lon 2D

    Reglas para standard_name:
      - si standard_name es distinto de None, se fuerza ese valor
      - si standard_name es None, se conserva el original del NetCDF si existe
      - además se guarda original_standard_name si estaba presente en origen
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

    if standard_name is None:
        resolved_standard_name = original_standard_name
    else:
        resolved_standard_name = standard_name

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
        raise ValueError("No hay datasets para comparar.")

    ref = datasets[0]
    ref_lat = ref["lat"].values
    ref_lon = ref["lon"].values
    ref_shape = ref[var_name].shape[1:]

    for i, ds in enumerate(datasets[1:], start=1):
        if ds[var_name].shape[1:] != ref_shape:
            raise ValueError(
                f"Shape espacial inconsistente en dataset {i}: "
                f"{ds[var_name].shape[1:]} != {ref_shape}"
            )
        if not np.allclose(ds["lat"].values, ref_lat, rtol=rtol, atol=atol, equal_nan=True):
            raise ValueError(f"lat inconsistente en dataset {i}")
        if not np.allclose(ds["lon"].values, ref_lon, rtol=rtol, atol=atol, equal_nan=True):
            raise ValueError(f"lon inconsistente en dataset {i}")


def add_crs_metadata(ds: xr.Dataset, epsg: str = "EPSG:4326") -> xr.Dataset:
    """
    Añade variable spatial_ref con WKT para compatibilidad.
    """
    crs = CRS.from_user_input(epsg)

    ds["spatial_ref"] = xr.DataArray(
        0,
        attrs={
            "spatial_ref": crs.to_wkt(),
            "crs_wkt": crs.to_wkt(),
            "grid_mapping_name": "latitude_longitude",
            "epsg_code": epsg,
            "semi_major_axis": crs.ellipsoid.semi_major_metre,
            "inverse_flattening": crs.ellipsoid.inverse_flattening,
        },
    )
    return ds


def load_and_standardize_nc_files(
    nc_files: Iterable[str | Path],
    var_name: str = DEFAULT_VAR_NAME,
    standard_name: str | None = DEFAULT_STANDARD_NAME,
    verbose: bool = True,
) -> list[xr.Dataset]:
    standardized: list[xr.Dataset] = []

    for fp in nc_files:
        fp = Path(fp)
        if verbose:
            print(f"Leyendo: {fp}")

        ds = xr.open_dataset(fp, decode_times=True, mask_and_scale=False)

        radar_var = infer_radar_variable(ds)
        inspect_raw_radar_dataset(ds, radar_var)

        try:
            if verbose:
                print(f"Variable radar inferida: {radar_var}")
                print(ds)
                print("data_vars:", list(ds.data_vars))
                for v in ds.variables:
                    print(v, ds[v].dims, ds[v].shape)

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
        raise ValueError("No hay datasets para concatenar.")

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
    license_name: str = "OGL-UK-3.0",
    mlcast_created_by: str = "Tu Nombre <tu_email@aemet.es>",
    mlcast_created_with: str = "https://github.com/tu-org/tu-repo@v0.1.0",
    mlcast_dataset_version: str = "0.1.0",
    mlcast_dataset_identifier: str = "es-aemet-radar-qa",
    mlcast_dataset_identifier_format: str = "{country_code}-{entity}-{physical_variable}-{time_resolution}-{common_name}",
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
            "mlcast_created_on": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "mlcast_created_by": mlcast_created_by,
            "mlcast_created_with": mlcast_created_with,
            "mlcast_dataset_version": mlcast_dataset_version,
            "mlcast_dataset_identifier": mlcast_dataset_identifier,
            "mlcast_dataset_identifier_format": mlcast_dataset_identifier_format,
        }
    )

    ds = ds.chunk({"time": 1, "y": -1, "x": -1})
    return ds


def _clean_variable_attrs_for_zarr(ds: xr.Dataset, var_name: str = DEFAULT_VAR_NAME) -> xr.Dataset:
    ds = ds.copy()

    for var in [var_name, "lat", "lon"]:
        if var in ds:
            for key in ["_FillValue", "missing_value", "fill_value", "add_offset", "scale_factor"]:
                ds[var].attrs.pop(key, None)
                ds[var].encoding.pop(key, None)

    return ds


def save_radar_dataset_to_zarr(
    ds: xr.Dataset,
    zarr_path: str | Path,
    var_name: str = DEFAULT_VAR_NAME,
    overwrite: bool = True,
    compressor_name: str = "zstd",
    compression_level: int = 5,
    blosc_shuffle: str = "bitshuffle",
) -> Path:
    zarr_path = Path(zarr_path)

    if overwrite and zarr_path.exists():
        shutil.rmtree(zarr_path)

    compressor = build_compressor(
        compressor_name=compressor_name,
        compression_level=compression_level,
        blosc_shuffle=blosc_shuffle,
    )

    encoding = {
        var_name: {
            "compressor": compressor,
            "dtype": "float32",
            "_FillValue": np.nan,
            "chunks": (1, ds.sizes["y"], ds.sizes["x"]),
        },
        "lat": {
            "compressor": compressor,
            "dtype": "float32",
        },
        "lon": {
            "compressor": compressor,
            "dtype": "float32",
        },
    }

    try:
        ds.to_zarr(
            zarr_path,
            mode="w",
            consolidated=True,
            encoding=encoding,
            zarr_version=2,
        )
    except TypeError:
        ds.to_zarr(
            zarr_path,
            mode="w",
            consolidated=True,
            encoding=encoding,
            zarr_format=2,
        )

    print(
        f"Zarr guardado en: {zarr_path} "
        f"(compressor={compressor_name}, level={compression_level})"
    )
    return zarr_path


def inspect_radar_state(
    obj: xr.Dataset | str | Path,
    label: str,
    var_name: str = DEFAULT_VAR_NAME,
    time_index: int = 0,
    decode_cf_for_zarr: bool = True,
) -> None:
    """
    Inspecciona el estado del campo principal en distintos puntos del pipeline.

    - Si obj es un xr.Dataset, inspecciona en memoria.
    - Si obj es una ruta, asume que es un Zarr y lo abre.
    """
    print(f"\n{'=' * 80}")
    print(f"INSPECCIÓN: {label}")
    print(f"{'=' * 80}")

    if isinstance(obj, xr.Dataset):
        ds = obj
        source = "dataset en memoria"
    else:
        ds = xr.open_zarr(obj, consolidated=True, decode_cf=decode_cf_for_zarr)
        source = f"zarr ({'decode_cf=True' if decode_cf_for_zarr else 'decode_cf=False'})"

    print("Fuente:", source)
    print(ds)

    if var_name not in ds:
        print(f"No existe variable '{var_name}'")
        return

    field = ds[var_name].isel(time=time_index).load().values
    finite = np.isfinite(field)

    print("\n--- ESTADÍSTICAS DEL CAMPO ---")
    print("shape:", field.shape)
    print("dtype:", field.dtype)
    print("n_total:", field.size)
    print("n_finite:", int(finite.sum()))
    print("n_nan:", int(np.isnan(field).sum()))

    print("\n--- ATTRS VARIABLE ---")
    print(ds[var_name].attrs)

    print("\n--- ENCODING VARIABLE ---")
    print(ds[var_name].encoding)

    if finite.any():
        vals = field[finite]
        print("\n--- RANGO DE VALORES FINITOS ---")
        print("min:", float(vals.min()))
        print("max:", float(vals.max()))
        print("mean:", float(vals.mean()))
        print("p01:", float(np.percentile(vals, 1)))
        print("p05:", float(np.percentile(vals, 5)))
        print("p50:", float(np.percentile(vals, 50)))
        print("p95:", float(np.percentile(vals, 95)))
        print("p99:", float(np.percentile(vals, 99)))
        print("sample únicos:", np.unique(vals)[:20])
    else:
        print("\nNo hay valores finitos en este estado.")


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
    license_name: str = "OGL-UK-3.0",
    mlcast_created_by: str = "Tu Nombre <tu_email@aemet.es>",
    mlcast_created_with: str = "https://github.com/tu-org/tu-repo@v0.1.0",
    mlcast_dataset_version: str = "0.1.0",
    mlcast_dataset_identifier: str = "es-aemet-radar-qa",
    mlcast_dataset_identifier_format: str = "{country_code}-{entity}-{physical_variable}-{time_resolution}-{common_name}",
    compressor_name: str = "zstd",
    compression_level: int = 5,
    blosc_shuffle: str = "bitshuffle",
) -> xr.Dataset:
    standardized = load_and_standardize_nc_files(
        nc_files,
        var_name=var_name,
        standard_name=standard_name,
        verbose=verbose,
    )

    if inspect:
        print(f"\nNúmero de datasets estandarizados: {len(standardized)}")
        for i, ds in enumerate(standardized[:2]):
            inspect_radar_state(ds, label=f"standardized[{i}]", var_name=var_name, time_index=0)

    ds_all = concat_radar_datasets(standardized, var_name=var_name)

    if inspect:
        inspect_radar_state(
            ds_all,
            label="después de concat_radar_datasets",
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
        mlcast_created_by=mlcast_created_by,
        mlcast_created_with=mlcast_created_with,
        mlcast_dataset_version=mlcast_dataset_version,
        mlcast_dataset_identifier=mlcast_dataset_identifier,
        mlcast_dataset_identifier_format=mlcast_dataset_identifier_format,
    )

    ds_all = _clean_variable_attrs_for_zarr(ds_all, var_name=var_name)

    if inspect:
        inspect_radar_state(
            ds_all,
            label="después de prepare_radar_dataset_for_zarr",
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
    )

    if inspect:
        inspect_radar_state(
            zarr_path,
            label="después de save_radar_dataset_to_zarr (reapertura decode_cf=False)",
            var_name=var_name,
            time_index=0,
            decode_cf_for_zarr=False,
        )
        inspect_radar_state(
            zarr_path,
            label="después de save_radar_dataset_to_zarr (reapertura decode_cf=True)",
            var_name=var_name,
            time_index=0,
            decode_cf_for_zarr=True,
        )

    return ds_all


def open_radar_zarr(zarr_path: str | Path) -> xr.Dataset:
    return xr.open_zarr(zarr_path, consolidated=True)


def quick_validate_radar_zarr(
    zarr_path: str | Path,
    var_name: str = DEFAULT_VAR_NAME,
) -> None:
    print("\n--- VALIDACIÓN RÁPIDA ---")
    ds = open_radar_zarr(zarr_path)

    print(ds)

    if var_name not in ds:
        raise ValueError(f"Falta variable '{var_name}'")

    if ds[var_name].dims != ("time", "y", "x"):
        raise ValueError(f"Dims incorrectas en {var_name}: {ds[var_name].dims}")

    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError("Faltan coords lat/lon")

    if ds["lat"].dims != ("y", "x"):
        raise ValueError(f"lat dims incorrectas: {ds['lat'].dims}")
    if ds["lon"].dims != ("y", "x"):
        raise ValueError(f"lon dims incorrectas: {ds['lon'].dims}")

    if "spatial_ref" not in ds:
        raise ValueError("Falta variable spatial_ref")

    wkt = ds["spatial_ref"].attrs.get("crs_wkt") or ds["spatial_ref"].attrs.get("spatial_ref")
    if wkt is None:
        raise ValueError("Falta crs_wkt/spatial_ref en spatial_ref")

    from pyproj import CRS
    import cartopy.crs as ccrs

    crs = CRS.from_wkt(wkt)
    print("CRS leído desde WKT:", crs)

    cartopy_crs = ccrs.PlateCarree()
    print("CRS cartopy usado para visualización:", cartopy_crs)

    chunks = ds[var_name].encoding.get("chunks")
    print(f"Chunks {var_name}:", chunks)

    nan_count = int(np.isnan(ds[var_name].values).sum())
    print("Número de NaNs:", nan_count)

    print("Validación rápida OK")


def inspect_raw_radar_dataset(ds: xr.Dataset, radar_var: str) -> None:
    da = ds[radar_var]

    print("\n--- INSPECCIÓN RAW NETCDF ---")
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
    print("raw finite:", int(finite.sum()), "de", vals.size)


def inspect_radar_dataset_in_memory(
    ds: xr.Dataset,
    var_name: str = DEFAULT_VAR_NAME,
    time_index: int = 0,
) -> None:
    field = ds[var_name].isel(time=time_index).compute().values
    finite = np.isfinite(field)

    print("\n--- INSPECCIÓN EN MEMORIA ---")
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


def inspect_radar_zarr_raw(
    zarr_path: str | Path,
    var_name: str = DEFAULT_VAR_NAME,
    time_index: int = 0,
) -> None:
    ds = xr.open_zarr(zarr_path, consolidated=True, decode_cf=False)
    field = ds[var_name].isel(time=time_index).load().values
    finite = np.isfinite(field)

    print("\n--- INSPECCIÓN ZARR RAW (decode_cf=False) ---")
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
) -> None:
    ds = open_radar_zarr(zarr_path)
    field = ds[var_name].isel(time=time_index).load().values

    finite = np.isfinite(field)
    n_total = field.size
    n_finite = int(finite.sum())
    n_nan = int(np.isnan(field).sum())

    print("\n--- INSPECCIÓN CAMPO ---")
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
        print("No hay valores finitos en el campo.")


def plot_radar_from_zarr(
    zarr_path: str | Path,
    var_name: str = DEFAULT_VAR_NAME,
    time_index: int = 0,
    output_png: str | Path | None = None,
    figsize: tuple[int, int] = (8, 6),
    cmap: str = "viridis",
    add_colorbar: bool = True,
) -> None:
    """
    Plot rápido sin cartopy, usando lon/lat 2D.
    """
    ds = open_radar_zarr(zarr_path)

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
        print(f"Figura guardada en: {output_png}")

    plt.show()


def plot_radar_from_zarr_cartopy(
    zarr_path: str | Path,
    var_name: str = DEFAULT_VAR_NAME,
    time_index: int = 0,
    output_png: str | Path | None = None,
    figsize: tuple[int, int] = (9, 7),
    cmap: str = "viridis",
    add_coastlines: bool = True,
) -> None:
    """
    Plot rápido con cartopy para inspección geográfica.
    """
    import cartopy.crs as ccrs

    ds = open_radar_zarr(zarr_path)

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
        print(f"Figura guardada en: {output_png}")

    plt.show()