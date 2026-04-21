from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests
import xarray as xr
import zarr

from .data_handler import download_radar_data
from .radar_inspection import (
    inspect_radar_dataset_in_memory,
    inspect_radar_state,
    inspect_raw_radar_dataset,
)
from .radar_processing import (
    _clean_variable_attrs_for_zarr,
    concat_radar_datasets,
    load_and_standardize_nc_files,
    prepare_radar_dataset_for_zarr,
)
from .utils import _open_local_store, clean_dir, find_nc_files


@dataclass(slots=True)
class RadarBuildConfigZarr3:
    workdir: Path = Path("./workdir_zarr3")
    zarr_out: Path = Path("./ES_AEMET_reflectivity_v3.zarr")

    fechaini: str = "20241001T000000"
    fechafin: str = "20241001T015900"

    imagen: str = "PPI"
    configuracion: str = "Z_005_240"
    radar: str = "ZAR"
    epsg: str = "EPSG:4326"

    var_name: str = "equivalent_reflectivity_factor"
    standard_name: str | None = "equivalent_reflectivity_factor"

    mlcast_created_by: str = "Adrián García <agarciaa@aemet.es>"
    mlcast_created_with: str = "https://github.com/mlcast-community/mlcast-dataset-ES-AEMET-reflectivity@v0.1.0"
    mlcast_dataset_version: str = "0.1.0"
    mlcast_dataset_identifier: str = "ES-AEMET-radar_reflectivity-ppi_ZAR"
    mlcast_dataset_identifier_format: str = "{country_code}-{entity}-{physical_variable}-{common_name}"

    license: str = "CC-BY-4.0"
    institution: str = "Agencia Estatal de Meteorología (AEMET)"
    source: str = "AEMET radar network"
    attribution: str = "Data provided by AEMET"

    compression_level: int = 5
    time_chunk: int = 1
    shard_time: int = 144

    inspect: bool = False

    max_retries_429: int = 8
    initial_wait_429: int = 15
    backoff_factor_429: float = 2.0
    max_wait_429: int = 300


def parse_aemet_datetime(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%dT%H%M%S")


def format_aemet_datetime(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%S")


def iter_hourly_windows(start: datetime, end: datetime):
    current = start
    while current < end:
        nxt = min(current + timedelta(hours=1), end)
        yield current, nxt
        current = nxt


def safe_remove_path(path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        return
    if path.is_file() or path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def is_http_429_error(exc: Exception) -> bool:
    if isinstance(exc, requests.exceptions.HTTPError):
        response = getattr(exc, "response", None)
        if response is not None and response.status_code == 429:
            return True

    msg = str(exc)
    return "429" in msg and "Too Many Requests" in msg


def download_chunk_with_retries(
    *,
    config: RadarBuildConfigZarr3,
    fechaini_chunk: str,
    fechafin_chunk: str,
    chunk_dir: Path,
) -> None:
    api_key = os.environ.get("AEMET_API_KEY")
    if not api_key:
        raise ValueError("Define la variable de entorno AEMET_API_KEY antes de ejecutar el pipeline.")

    attempt = 0
    wait_seconds = config.initial_wait_429

    while True:
        try:
            download_radar_data(
                FECHAINI=fechaini_chunk,
                FECHAFIN=fechafin_chunk,
                IMAGEN=config.imagen,
                CONFIGURACION=config.configuracion,
                RADAR=config.radar,
                FORMATO="netcdf",
                EPSG=config.epsg,
                out_dir=chunk_dir,
                api_key=api_key,
            )
            return
        except Exception as exc:
            if not is_http_429_error(exc):
                raise
            attempt += 1
            if attempt > config.max_retries_429:
                raise RuntimeError(
                    f"Se superó el número máximo de reintentos por error 429 para el chunk {fechaini_chunk} -> {fechafin_chunk}"
                ) from exc
            print(
                f"[WARNING] HTTP 429 en chunk {fechaini_chunk} -> {fechafin_chunk}. "
                f"Reintento {attempt}/{config.max_retries_429} tras esperar {wait_seconds} s..."
            )
            time.sleep(wait_seconds)
            wait_seconds = min(int(wait_seconds * config.backoff_factor_429), config.max_wait_429)


def build_zarr_v3_compressors(compression_level: int = 5):
    """
    Compresores compatibles con zarr 3.1.6.
    """
    try:
        from zarr.codecs import ZstdCodec
    except Exception as e:
        raise RuntimeError(
            "No se pudo importar zarr.codecs.ZstdCodec. "
            "Revisa la instalación de zarr 3.x."
        ) from e

    return (ZstdCodec(level=compression_level),)


def _datetime64_to_int64(values: np.ndarray) -> np.ndarray:
    return values.astype("datetime64[ns]").astype(np.int64)


def _deduplicate_chunk_times(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.indexes:
        return ds
    return ds.sel(time=~ds.indexes["time"].duplicated())


def _filter_unseen_times(ds: xr.Dataset, seen_times: set[int]) -> xr.Dataset:
    if ds.sizes.get("time", 0) == 0:
        return ds

    time_int = _datetime64_to_int64(ds["time"].values)
    keep_mask = np.array([t not in seen_times for t in time_int], dtype=bool)
    if keep_mask.all():
        return ds
    return ds.isel(time=keep_mask)


def _initialize_v3_store_from_template(
    ds_template: xr.Dataset,
    zarr_path: str | Path,
    *,
    var_name: str,
    compression_level: int,
    time_chunk: int,
    shard_time: int,
    license_name: str,
    institution: str,
    source: str,
    attribution: str,
) -> tuple[zarr.Group, zarr.Array, zarr.Array]:
    zarr_path = Path(zarr_path)
    store = _open_local_store(zarr_path)
    root = zarr.group(store=store, overwrite=True, zarr_format=3)
    
    root.attrs.update(dict(ds_template.attrs))
    
    # license metadata
    root.attrs.update({
        "license": license_name,
        "institution": institution,
        "source": source,
        "attribution": attribution,
    })

    compressors = build_zarr_v3_compressors(compression_level=compression_level)

    arr_y = root.create_array(
        "y",
        shape=ds_template["y"].shape,
        dtype=ds_template["y"].dtype,
        chunks=ds_template["y"].shape,
        compressors=compressors,
        dimension_names=("y",),
        overwrite=True,
    )
    arr_x = root.create_array(
        "x",
        shape=ds_template["x"].shape,
        dtype=ds_template["x"].dtype,
        chunks=ds_template["x"].shape,
        compressors=compressors,
        dimension_names=("x",),
        overwrite=True,
    )
    arr_lat = root.create_array(
        "lat",
        shape=ds_template["lat"].shape,
        dtype="float32",
        chunks=ds_template["lat"].shape,
        compressors=compressors,
        dimension_names=("y", "x"),
        overwrite=True,
    )
    arr_lon = root.create_array(
        "lon",
        shape=ds_template["lon"].shape,
        dtype="float32",
        chunks=ds_template["lon"].shape,
        compressors=compressors,
        dimension_names=("y", "x"),
        overwrite=True,
    )
    arr_spatial_ref = root.create_array(
        "spatial_ref",
        shape=(),
        dtype=ds_template["spatial_ref"].dtype,
        chunks=(),
        dimension_names=(),
        overwrite=True,
    )

    arr_y[:] = ds_template["y"].values
    arr_x[:] = ds_template["x"].values
    arr_lat[:] = ds_template["lat"].values.astype("float32")
    arr_lon[:] = ds_template["lon"].values.astype("float32")
    arr_spatial_ref[()] = ds_template["spatial_ref"].values

    arr_y.attrs.update(dict(ds_template["y"].attrs))
    arr_x.attrs.update(dict(ds_template["x"].attrs))
    arr_lat.attrs.update(dict(ds_template["lat"].attrs))
    arr_lon.attrs.update(dict(ds_template["lon"].attrs))
    arr_spatial_ref.attrs.update(dict(ds_template["spatial_ref"].attrs))

    arr_time = root.create_array(
        "time",
        shape=(0,),
        dtype=ds_template["time"].dtype,
        chunks=(max(shard_time, 1),),
        compressors=compressors,
        dimension_names=("time",),
        overwrite=True,
    )
    arr_time.attrs.update(dict(ds_template["time"].attrs))

    arr_radar = root.create_array(
        var_name,
        shape=(0, ds_template.sizes["y"], ds_template.sizes["x"]),
        dtype="float32",
        chunks=(time_chunk, ds_template.sizes["y"], ds_template.sizes["x"]),
        shards=(shard_time, ds_template.sizes["y"], ds_template.sizes["x"]),
        fill_value=np.nan,
        compressors=compressors,
        dimension_names=("time", "y", "x"),
        overwrite=True,
    )
    arr_radar.attrs.update(dict(ds_template[var_name].attrs))

    return root, arr_time, arr_radar


def _append_ds_to_v3_store(
    ds_chunk: xr.Dataset,
    *,
    arr_time: zarr.Array,
    arr_radar: zarr.Array,
    var_name: str,
) -> int:
    n_new = ds_chunk.sizes.get("time", 0)
    if n_new == 0:
        return 0

    time_vals = ds_chunk["time"].values
    radar_vals = ds_chunk[var_name].values.astype("float32")

    arr_time.append(time_vals, axis=0)
    arr_radar.append(radar_vals, axis=0)
    return n_new


def run_pipeline_zarr3(config: RadarBuildConfigZarr3) -> None:
    config.workdir = Path(config.workdir)
    config.zarr_out = Path(config.zarr_out)

    if config.time_chunk != 1:
        print("[WARNING] time_chunk != 1. Eso ya no cumple '1 chunk per timestep'.")
    if config.shard_time < config.time_chunk:
        raise ValueError("shard_time debe ser >= time_chunk")

    clean_dir(config.workdir)
    safe_remove_path(config.zarr_out)

    dt_ini = parse_aemet_datetime(config.fechaini)
    dt_fin = parse_aemet_datetime(config.fechafin)
    if dt_fin <= dt_ini:
        raise ValueError("fechafin debe ser posterior a fechaini.")

    root = None
    arr_time = None
    arr_radar = None
    seen_times: set[int] = set()

    n_chunks_ok = 0
    n_chunks_empty = 0
    n_chunks_failed = 0
    n_steps_written = 0

    for i, (w_ini, w_fin) in enumerate(iter_hourly_windows(dt_ini, dt_fin), start=1):
        w_ini_str = format_aemet_datetime(w_ini)
        w_fin_str = format_aemet_datetime(w_fin)

        chunk_dir = config.workdir / f"chunk_{i:04d}_{w_ini_str}_{w_fin_str}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 100)
        print(f"CHUNK {i}")
        print(f"Ventana: {w_ini_str} -> {w_fin_str}")
        print(f"Directorio temporal: {chunk_dir}")
        print("=" * 100)

        try:
            download_chunk_with_retries(
                config=config,
                fechaini_chunk=w_ini_str,
                fechafin_chunk=w_fin_str,
                chunk_dir=chunk_dir,
            )
            nc_files_chunk = sorted(find_nc_files(chunk_dir))
            print(f"Se han encontrado {len(nc_files_chunk)} archivos .nc en este chunk")

            if not nc_files_chunk:
                print(f"[WARNING] Chunk vacío: {w_ini_str} -> {w_fin_str}")
                n_chunks_empty += 1
                continue

            standardized = load_and_standardize_nc_files(
                nc_files_chunk,
                var_name=config.var_name,
                standard_name=config.standard_name,
                verbose=False,
                inspect_raw_fn=inspect_raw_radar_dataset if config.inspect else None,
            )
            if not standardized:
                print(f"[WARNING] Chunk sin datasets estandarizados: {w_ini_str} -> {w_fin_str}")
                n_chunks_empty += 1
                continue

            ds_chunk = concat_radar_datasets(standardized, var_name=config.var_name)
            ds_chunk = prepare_radar_dataset_for_zarr(
                ds_chunk,
                var_name=config.var_name,
                epsg=config.epsg,
                institution=config.institution,
                source=config.source,
                license_name=config.license,
                attribution=config.attribution,
                mlcast_created_by=config.mlcast_created_by,
                mlcast_created_with=config.mlcast_created_with,
                mlcast_dataset_version=config.mlcast_dataset_version,
                mlcast_dataset_identifier=config.mlcast_dataset_identifier,
                mlcast_dataset_identifier_format=config.mlcast_dataset_identifier_format,
                time_chunk=config.time_chunk,
            )
            ds_chunk = _clean_variable_attrs_for_zarr(ds_chunk, var_name=config.var_name)
            ds_chunk = _deduplicate_chunk_times(ds_chunk)
            ds_chunk = _filter_unseen_times(ds_chunk, seen_times)

            if ds_chunk.sizes.get("time", 0) == 0:
                print(f"[WARNING] Chunk sin tiempos nuevos tras deduplicación: {w_ini_str} -> {w_fin_str}")
                n_chunks_empty += 1
                continue

            if config.inspect:
                inspect_radar_state(
                    ds_chunk,
                    label=f"chunk preparado [{w_ini_str} -> {w_fin_str}]",
                    var_name=config.var_name,
                    time_index=0,
                )

            if root is None or arr_time is None or arr_radar is None:
                root, arr_time, arr_radar = _initialize_v3_store_from_template(
                    ds_chunk,
                    config.zarr_out,
                    var_name=config.var_name,
                    compression_level=config.compression_level,
                    time_chunk=config.time_chunk,
                    shard_time=config.shard_time,
                    license_name=config.license,
                    institution=config.institution,
                    source=config.source,
                    attribution=config.attribution,
                )
                print(f"[OK] Inicializado store Zarr v3 sharded: {config.zarr_out}")

            n_written_chunk = _append_ds_to_v3_store(
                ds_chunk,
                arr_time=arr_time,
                arr_radar=arr_radar,
                var_name=config.var_name,
            )
            seen_times.update(_datetime64_to_int64(ds_chunk["time"].values).tolist())
            n_steps_written += n_written_chunk
            n_chunks_ok += 1
            print(f"[OK] Añadidos {n_written_chunk} timesteps al store final")

        except Exception as exc:
            n_chunks_failed += 1
            print(f"[ERROR] Falló el chunk {i} ({w_ini_str} -> {w_fin_str}): {exc}")
        finally:
            safe_remove_path(chunk_dir)

    print("\n" + "#" * 100)
    print("RESUMEN")
    print("#" * 100)
    print(f"Chunks correctos: {n_chunks_ok}")
    print(f"Chunks vacíos:    {n_chunks_empty}")
    print(f"Chunks fallidos:  {n_chunks_failed}")
    print(f"Timesteps escritos: {n_steps_written}")

    if arr_time is None or arr_radar is None:
        raise RuntimeError("No se ha podido construir el Zarr final: no hubo ningún chunk válido con datos.")

    ds = xr.open_zarr(config.zarr_out, consolidated=False)
    print("\nDataset final:")
    print(ds)
    if ds.sizes.get("time", 0) > 0:
        inspect_radar_dataset_in_memory(ds, var_name=config.var_name, time_index=0)
    ds.close()