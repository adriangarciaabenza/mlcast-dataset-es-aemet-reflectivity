from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr
import zarr

from mlcast_dataset_es_aemet_reflectivity.pipeline_zarr3 import (
    _initialize_v3_store_from_template,
    _append_ds_to_v3_store,
)


def open_monthly_zarr(path: str | Path) -> xr.Dataset:
    ds = xr.open_zarr(path, consolidated=False)
    return ds


def deduplicate_times(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.indexes:
        return ds
    return ds.sel(time=~ds.indexes["time"].duplicated())


def filter_unseen_times(ds: xr.Dataset, seen_times: set[int]) -> xr.Dataset:
    if ds.sizes.get("time", 0) == 0:
        return ds
    time_int = ds["time"].values.astype("datetime64[ns]").astype(np.int64)
    keep_mask = np.array([t not in seen_times for t in time_int], dtype=bool)
    if keep_mask.all():
        return ds
    return ds.isel(time=keep_mask)


def merge_monthly_zarrs(
    monthly_paths: Iterable[str | Path],
    output_zarr: str | Path,
    *,
    var_name: str = "equivalent_reflectivity_factor",
    compression_level: int = 5,
    time_chunk: int = 1,
    shard_time: int = 144,
    license_name: str = "CC-BY-4.0",
    institution: str = "Agencia Estatal de Meteorología (AEMET)",
    source: str = "AEMET radar network",
    attribution: str = "Data provided by AEMET",
) -> None:
    monthly_paths = [Path(p) for p in monthly_paths]
    monthly_paths = sorted(monthly_paths)

    output_zarr = Path(output_zarr)
    if output_zarr.exists():
        import shutil
        shutil.rmtree(output_zarr)

    root = None
    arr_time = None
    arr_radar = None
    seen_times: set[int] = set()

    for path in monthly_paths:
        print(f"Procesando: {path}")
        ds = open_monthly_zarr(path)
        try:
            ds = deduplicate_times(ds)
            ds = filter_unseen_times(ds, seen_times)

            if ds.sizes.get("time", 0) == 0:
                print(f"[WARNING] Sin tiempos nuevos en {path}")
                continue

            if root is None or arr_time is None or arr_radar is None:
                root, arr_time, arr_radar = _initialize_v3_store_from_template(
                    ds,
                    output_zarr,
                    var_name=var_name,
                    compression_level=compression_level,
                    time_chunk=time_chunk,
                    shard_time=shard_time,
                    license_name=license_name,
                    institution=institution,
                    source=source,
                    attribution=attribution,
                )
                print(f"[OK] Inicializado Zarr final: {output_zarr}")

            n_written = _append_ds_to_v3_store(
                ds,
                arr_time=arr_time,
                arr_radar=arr_radar,
                var_name=var_name,
            )
            seen_times.update(ds["time"].values.astype("datetime64[ns]").astype(np.int64).tolist())
            print(f"[OK] Añadidos {n_written} timesteps desde {path}")
        finally:
            ds.close()

    if arr_time is None or arr_radar is None:
        raise RuntimeError("No se escribió ningún mes en el Zarr final.")

    ds_final = xr.open_zarr(output_zarr, consolidated=False)
    try:
        print(ds_final)
    finally:
        ds_final.close()


if __name__ == "__main__":
    monthly_dir = Path("/lustre/utmp/std/MLCAST_radar_data/outputs/monthly_v3")
    monthly_paths = sorted(monthly_dir.glob("radar_*.zarr"))

    merge_monthly_zarrs(
        monthly_paths=monthly_paths,
        output_zarr="/perm/pred/std/ML/MLCAST/mlcast-dataset-ES-AEMET-reflectivity/ES-AEMET-radar_reflectivity-ppi_ZAR_2020-2023.zarr",
        var_name="equivalent_reflectivity_factor",
        compression_level=5,
        time_chunk=1,
        shard_time=144,
        license_name="CC-BY-4.0",
        institution="Agencia Estatal de Meteorología (AEMET)",
        source="AEMET radar network",
        attribution="Data provided by AEMET",
    )