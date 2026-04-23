from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import requests
import xarray as xr

from .data_handler import download_radar_data
from .utils import clean_dir, find_nc_files
from .radar_processing import build_radar_zarr_from_nc_files
from .radar_inspection import (
    inspect_radar_dataset_in_memory,
    inspect_raw_radar_dataset,
    inspect_radar_state,
)


@dataclass(slots=True)
class RadarBuildConfig:
    workdir: Path = Path("./workdir")
    zarr_out: Path = Path("./ES_AEMET_reflectivity.zarr")
    png_out: Path = Path("./radar_quicklook.png")
    png_out_cartopy: Path = Path("./radar_quicklook_cartopy.png")

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

    compressor_name: str = "zstd"
    compression_level: int = 5
    blosc_shuffle: str = "bitshuffle"

    zarr_format: int = 2
    time_chunk: int = 1
    use_sharding: bool = False
    shard_time: int | None = None

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


def write_ds_to_zarr_v2(
    ds: xr.Dataset,
    store: str | Path,
    mode: str,
    append_dim: str | None = None,
) -> None:
    store = Path(store)
    kwargs = {"store": store, "mode": mode}
    if append_dim is not None:
        kwargs["append_dim"] = append_dim

    try:
        ds.to_zarr(**kwargs, zarr_version=2)
    except TypeError:
        ds.to_zarr(**kwargs, zarr_format=2)


def append_chunk_zarr_to_final(
    chunk_zarr_path: str | Path,
    final_zarr_path: str | Path,
    first_chunk: bool,
) -> bool:
    chunk_zarr_path = Path(chunk_zarr_path)
    final_zarr_path = Path(final_zarr_path)

    ds_chunk = xr.open_zarr(chunk_zarr_path, consolidated=True)
    ds_chunk = ds_chunk.load()

    if "time" in ds_chunk.indexes:
        ds_chunk = ds_chunk.sel(time=~ds_chunk.indexes["time"].duplicated())

    try:
        if first_chunk:
            write_ds_to_zarr_v2(ds_chunk, final_zarr_path, mode="w")
            print(f"[OK] Final Zarr store created from the first chunk: {final_zarr_path}")
            return False

        write_ds_to_zarr_v2(ds_chunk, final_zarr_path, mode="a", append_dim="time")
        print(f"[OK] Chunk appended to the final Zarr store: {final_zarr_path}")
        return False
    finally:
        ds_chunk.close()


def is_http_429_error(exc: Exception) -> bool:
    if isinstance(exc, requests.exceptions.HTTPError):
        response = getattr(exc, "response", None)
        if response is not None and response.status_code == 429:
            return True

    msg = str(exc)
    return "429" in msg and "Too Many Requests" in msg


def download_chunk_with_retries(*, config: RadarBuildConfig, fechaini_chunk: str, fechafin_chunk: str, chunk_dir: Path) -> None:
    api_key = os.environ.get("AEMET_API_KEY")
    if not api_key:
        raise ValueError("Define the AEMET_API_KEY environment variable before running the pipeline.")

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
                    f"Maximum number of retries exceeded due to HTTP 429 for chunk {fechaini_chunk} -> {fechafin_chunk}"
                ) from exc
            print(
                f"[WARNING] HTTP 429 on chunk {fechaini_chunk} -> {fechafin_chunk}. "
                f"Retry {attempt}/{config.max_retries_429} after waiting {wait_seconds} s..."
            )
            time.sleep(wait_seconds)
            wait_seconds = min(int(wait_seconds * config.backoff_factor_429), config.max_wait_429)


def run_pipeline(config: RadarBuildConfig) -> None:
    config.workdir = Path(config.workdir)
    config.zarr_out = Path(config.zarr_out)
    config.png_out = Path(config.png_out)
    config.png_out_cartopy = Path(config.png_out_cartopy)

    clean_dir(config.workdir)
    safe_remove_path(config.zarr_out)
    safe_remove_path(config.png_out)
    safe_remove_path(config.png_out_cartopy)

    dt_ini = parse_aemet_datetime(config.fechaini)
    dt_fin = parse_aemet_datetime(config.fechafin)
    if dt_fin <= dt_ini:
        raise ValueError("fechafin must be later than fechaini.")

    if config.zarr_format != 2:
        raise NotImplementedError(
            "This incremental append-based pipeline is designed for Zarr v2. "
            "For v3 with sharding, it is better to write the final store directly."
        )

    first_chunk = True
    n_chunks_ok = 0
    n_chunks_empty = 0
    n_chunks_failed = 0

    for i, (w_ini, w_fin) in enumerate(iter_hourly_windows(dt_ini, dt_fin), start=1):
        w_ini_str = format_aemet_datetime(w_ini)
        w_fin_str = format_aemet_datetime(w_fin)

        chunk_dir = config.workdir / f"chunk_{i:04d}_{w_ini_str}_{w_fin_str}"
        chunk_zarr = config.workdir / f"chunk_{i:04d}_{w_ini_str}_{w_fin_str}.zarr"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 100)
        print(f"CHUNK {i}")
        print(f"Window: {w_ini_str} -> {w_fin_str}")
        print(f"Temporary directory: {chunk_dir}")
        print(f"Temporary Zarr: {chunk_zarr}")
        print("=" * 100)

        try:
            download_chunk_with_retries(
                config=config,
                fechaini_chunk=w_ini_str,
                fechafin_chunk=w_fin_str,
                chunk_dir=chunk_dir,
            )
            nc_files_chunk = sorted(find_nc_files(chunk_dir))
            print(f"{len(nc_files_chunk)} .nc files were found in this chunk")

            if not nc_files_chunk:
                print(f"[WARNING] Empty chunk: {w_ini_str} -> {w_fin_str}")
                n_chunks_empty += 1
                continue

            build_radar_zarr_from_nc_files(
                nc_files=nc_files_chunk,
                zarr_path=chunk_zarr,
                var_name=config.var_name,
                standard_name=config.standard_name,
                epsg=config.epsg,
                overwrite=True,
                mlcast_created_by=config.mlcast_created_by,
                mlcast_created_with=config.mlcast_created_with,
                mlcast_dataset_version=config.mlcast_dataset_version,
                mlcast_dataset_identifier=config.mlcast_dataset_identifier,
                mlcast_dataset_identifier_format=config.mlcast_dataset_identifier_format,
                compressor_name=config.compressor_name,
                compression_level=config.compression_level,
                blosc_shuffle=config.blosc_shuffle,
                zarr_format=config.zarr_format,
                time_chunk=config.time_chunk,
                use_sharding=config.use_sharding,
                shard_time=config.shard_time,
                inspect=False,
                inspect_raw_fn=inspect_raw_radar_dataset,
                inspect_state_fn=inspect_radar_state,
            )

            first_chunk = append_chunk_zarr_to_final(
                chunk_zarr_path=chunk_zarr,
                final_zarr_path=config.zarr_out,
                first_chunk=first_chunk,
            )
            n_chunks_ok += 1
        except Exception as exc:
            n_chunks_failed += 1
            print(f"[ERROR] Chunk {i} failed ({w_ini_str} -> {w_fin_str}): {exc}")
        finally:
            safe_remove_path(chunk_dir)
            safe_remove_path(chunk_zarr)

    print("\n" + "#" * 100)
    print("SUMMARY")
    print("#" * 100)
    print(f"Successful chunks: {n_chunks_ok}")
    print(f"Empty chunks:      {n_chunks_empty}")
    print(f"Failed chunks:     {n_chunks_failed}")

    if first_chunk:
        raise RuntimeError("Could not build the final Zarr store: there was no valid chunk with data.")

    ds = xr.open_zarr(config.zarr_out, consolidated=False)
    print("\nFinal dataset:")
    print(ds)
    if ds.sizes.get("time", 0) > 0:
        inspect_radar_dataset_in_memory(ds, var_name=config.var_name, time_index=0)
    ds.close()