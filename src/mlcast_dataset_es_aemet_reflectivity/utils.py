from __future__ import annotations

import shutil
from pathlib import Path

import xarray as xr
import zarr
from numcodecs import Blosc, Zstd
from pyproj import CRS


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


def _rename_only_existing_dims(da, rename_map: dict[str, str]):
    """
    Rename only the dimensions that actually exist in the DataArray.
    """
    local_map = {d: rename_map[d] for d in da.dims if d in rename_map}
    if local_map:
        da = da.rename(local_map)
    return da


def _copy_attrs_safe(src, dst):
    dst.attrs.update(dict(src.attrs))
    return dst


def build_compressor(
    compressor_name: str = "zstd",
    compression_level: int = 5,
    blosc_shuffle: str = "bitshuffle",
):
    """
    Build a configurable numcodecs compressor for Zarr v2.
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
                f"Invalid blosc_shuffle: {blosc_shuffle}. "
                "Use 'bitshuffle', 'shuffle' or 'noshuffle'."
            )

        return Blosc(
            cname="zstd",
            clevel=compression_level,
            shuffle=shuffle_value,
        )

    raise ValueError(
        f"Invalid compressor_name: {compressor_name}. "
        "Use 'zstd' or 'blosc'."
    )


def add_crs_metadata(ds, epsg: str = "EPSG:4326"):
    """
    Add a spatial_ref variable with WKT for compatibility.
    """
    crs = CRS.from_user_input(epsg)

    attrs = {
        "spatial_ref": crs.to_wkt(),
        "crs_wkt": crs.to_wkt(),
        "epsg_code": epsg,
    }

    if crs.is_projected:
        attrs["grid_mapping_name"] = "transverse_mercator"
    else:
        attrs["grid_mapping_name"] = "latitude_longitude"
        attrs["semi_major_axis"] = crs.ellipsoid.semi_major_metre
        attrs["inverse_flattening"] = crs.ellipsoid.inverse_flattening

    ds["spatial_ref"] = xr.DataArray(0, attrs=attrs)
    return ds


def _open_local_store(path: str | Path):
    """
    Try to open a valid local store for zarr-python 3.x.
    """
    path = str(path)

    try:
        return zarr.storage.LocalStore(path)
    except Exception:
        pass

    try:
        return zarr.storage.DirectoryStore(path)
    except Exception as e:
        raise RuntimeError(
            "Could not create a local store for Zarr. "
            "Check the installed zarr-python version."
        ) from e