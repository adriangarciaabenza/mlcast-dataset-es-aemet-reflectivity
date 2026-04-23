from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from mlcast_dataset_es_aemet_reflectivity.radar_inspection import (
    inspect_radar_dataset_in_memory,
)

# =============================================================================
# CONFIG
# =============================================================================

# Cambia esta ruta a tu store Zarr
ZARR_PATH = Path("./ES-AEMET-radar_reflectivity-ppi_ZAR_2020-2024.zarr/")

# Variable principal a explorar
VAR_NAME = "equivalent_reflectivity_factor"

# Índice temporal de ejemplo
TIME_INDEX = 0

# Si quieres seleccionar por valor en vez de por índice
USE_SEL_INSTEAD_OF_ISEL = False
TIME_VALUE = None  # por ejemplo: "2024-10-01T00:00:00"


# =============================================================================
# HELPERS
# =============================================================================


def detect_zarr_format(path: Path) -> int:
    """
    Detecta el formato Zarr de forma simple:
    - v3 si existe zarr.json en raíz
    - v2 en caso contrario
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
        raise FileNotFoundError(f"No existe el store Zarr: {path}")

    if zarr_format == 3:
        ds = xr.open_zarr(path, consolidated=None, zarr_format=3)
    else:
        ds = xr.open_zarr(path, consolidated=True)

    return ds


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    ds = open_zarr_dataset(ZARR_PATH)

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
        #print(f"{v}: shape={ds[v].shape}, dtype={ds[v].dtype}, chunks={chunks}")

    if VAR_NAME not in ds:
        raise KeyError(f"No existe la variable '{VAR_NAME}' en el dataset")

    da = ds[VAR_NAME]

    print("\n=== SELECTED VARIABLE ===")
    print(da)
    print("\n=== ATTRS ===")
    print(da.attrs)
    print("\n=== ENCODING ===")
    print(da.encoding)

    # Inspección rápida del primer campo temporal
    if da.sizes.get("time", 0) > 0:
        print("\n=== QUICK IN-MEMORY INSPECTION ===")
        inspect_radar_dataset_in_memory(ds, var_name=VAR_NAME, time_index=TIME_INDEX)

    # Selección de un campo 2D
    print("\n=== SELECTING 2D FIELD ===")
    if USE_SEL_INSTEAD_OF_ISEL:
        if TIME_VALUE is None:
            raise ValueError("Si USE_SEL_INSTEAD_OF_ISEL=True, TIME_VALUE no puede ser None")
        field = da.sel(time=TIME_VALUE)
    else:
        field = da.isel(time=TIME_INDEX)

    print(field)

    print("\n=== LOADING ONLY THIS 2D FIELD ===")
    field_loaded = field.compute()
    print(field_loaded)

    # Plot simple usando lat/lon si existen
    plt.figure(figsize=(10, 6))
    if "lat" in ds.coords and "lon" in ds.coords and ds["lat"].ndim == 2 and ds["lon"].ndim == 2:
        plt.pcolormesh(ds["lon"].values, ds["lat"].values, field_loaded.values, shading="auto")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.colorbar(label=field_loaded.attrs.get("units", ""))
    else:
        field_loaded.plot(cmap="viridis")

    title_time = getattr(field_loaded, "time", None)
    if title_time is not None:
        plt.title(f"{VAR_NAME} | time={field_loaded.time.values}")
    else:
        plt.title(VAR_NAME)

    plt.tight_layout()
    plt.show()

    ds.close()


if __name__ == "__main__":
    main()
