from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar


DEFAULT_INPUT_ZARR = Path(
    "/lustre/utmp/std/MLCAST_radar_data/"
    "ES-AEMET-rainfall_rate-sri_ZAR_2020-2024_v1.zarr"
)

DEFAULT_OUTPUT_ZARR = Path(
    "/lustre/utmp/std/MLCAST_radar_data/"
    "ES-AEMET-rainfall_rate-sri_ZAR_2020-2024_v1_converted.zarr"
)

DEFAULT_VAR_NAME = "rainfall_rate"
DEFAULT_MARSHALL_PALMER_A = 200.0
DEFAULT_MARSHALL_PALMER_B = 1.6
DEFAULT_WATER_DENSITY_KG_M3 = 1000.0
DEFAULT_TIME_SHARD = 144
DEFAULT_OUTPUT_MODE = "final-only"


def detect_zarr_format(path: Path) -> int:
    if (path / "zarr.json").exists():
        return 3
    return 2


def dbz_to_rainfall_rate_mm_h(
    dbz: xr.DataArray,
    marshall_palmer_a: float,
    marshall_palmer_b: float,
) -> xr.DataArray:
    z_linear = 10.0 ** (dbz / 10.0)
    return (z_linear / marshall_palmer_a) ** (1.0 / marshall_palmer_b)


def rainfall_rate_mm_h_to_kg_m2_h(
    rainfall_rate_mm_h: xr.DataArray,
    water_density_kg_m3: float,
) -> xr.DataArray:
    return rainfall_rate_mm_h * (water_density_kg_m3 / 1000.0)


def chunk_dataarray_like_mlcast(da: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
    return da.chunk(
        {
            "time": 1,
            "y": ds.sizes["y"],
            "x": ds.sizes["x"],
        }
    )


def build_final_only_dataset(
    ds: xr.Dataset,
    variable_name: str,
    original_attrs: dict,
    marshall_palmer_a: float,
    marshall_palmer_b: float,
    water_density_kg_m3: float,
) -> xr.Dataset:
    rainfall_rate_mm_h = dbz_to_rainfall_rate_mm_h(
        ds[variable_name],
        marshall_palmer_a=marshall_palmer_a,
        marshall_palmer_b=marshall_palmer_b,
    )

    rainfall_flux_kg_m2_h = rainfall_rate_mm_h_to_kg_m2_h(
        rainfall_rate_mm_h,
        water_density_kg_m3=water_density_kg_m3,
    )

    ds[variable_name] = chunk_dataarray_like_mlcast(
        rainfall_flux_kg_m2_h.astype("float32"),
        ds,
    )

    ds[variable_name].attrs = {
        "long_name": (
            "Rainfall mass flux derived from radar reflectivity using the "
            "Marshall-Palmer relation"
        ),
        "standard_name": "rainfall_flux",
        "units": "kg m-2 h-1",
        "grid_mapping": original_attrs.get("grid_mapping", "spatial_ref"),
        "source_units": "dBZ",
        "original_units_before_conversion": original_attrs.get("units", "unknown"),
        "conversion_formula": (
            "Z = a * R**b; R = (10**(dBZ / 10) / a)**(1 / b); "
            "rainfall_flux = R * rho_water / 1000"
        ),
        "marshall_palmer_a": marshall_palmer_a,
        "marshall_palmer_b": marshall_palmer_b,
        "water_density_kg_m3": water_density_kg_m3,
    }

    return ds


def build_full_dataset(
    ds: xr.Dataset,
    variable_name: str,
    original_attrs: dict,
    marshall_palmer_a: float,
    marshall_palmer_b: float,
    water_density_kg_m3: float,
) -> xr.Dataset:
    dbz_name = f"{variable_name}_dbz"
    mm_h_name = f"{variable_name}_mm_h"

    ds = ds.rename({variable_name: dbz_name})

    ds[dbz_name] = chunk_dataarray_like_mlcast(ds[dbz_name], ds)

    ds[dbz_name].attrs = {
        **original_attrs,
        "long_name": "Radar reflectivity factor in logarithmic units",
        "standard_name": "equivalent_reflectivity_factor",
        "units": "dBZ",
        "original_units_before_conversion": original_attrs.get("units", "unknown"),
        "conversion_note": (
            "This variable is assumed to contain dBZ values, despite previous "
            "metadata possibly indicating rainfall-rate units."
        ),
    }

    rainfall_rate_mm_h = dbz_to_rainfall_rate_mm_h(
        ds[dbz_name],
        marshall_palmer_a=marshall_palmer_a,
        marshall_palmer_b=marshall_palmer_b,
    )

    rainfall_flux_kg_m2_h = rainfall_rate_mm_h_to_kg_m2_h(
        rainfall_rate_mm_h,
        water_density_kg_m3=water_density_kg_m3,
    )

    ds[mm_h_name] = chunk_dataarray_like_mlcast(
        rainfall_rate_mm_h.astype("float32"),
        ds,
    )

    ds[variable_name] = chunk_dataarray_like_mlcast(
        rainfall_flux_kg_m2_h.astype("float32"),
        ds,
    )

    ds[mm_h_name].attrs = {
        "long_name": (
            "Rainfall rate derived from radar reflectivity using the "
            "Marshall-Palmer relation"
        ),
        "standard_name": "rainfall_rate",
        "units": "mm h-1",
        "grid_mapping": original_attrs.get("grid_mapping", "spatial_ref"),
        "source_variable": dbz_name,
        "conversion_formula": "Z = a * R**b; R = (10**(dBZ / 10) / a)**(1 / b)",
        "marshall_palmer_a": marshall_palmer_a,
        "marshall_palmer_b": marshall_palmer_b,
    }

    ds[variable_name].attrs = {
        "long_name": (
            "Rainfall mass flux derived from radar reflectivity using the "
            "Marshall-Palmer relation"
        ),
        "standard_name": "rainfall_flux",
        "units": "kg m-2 h-1",
        "grid_mapping": original_attrs.get("grid_mapping", "spatial_ref"),
        "source_variable": dbz_name,
        "intermediate_variable": mm_h_name,
        "conversion_formula": (
            "Z = a * R**b; R = (10**(dBZ / 10) / a)**(1 / b); "
            "rainfall_flux = R * rho_water / 1000"
        ),
        "marshall_palmer_a": marshall_palmer_a,
        "marshall_palmer_b": marshall_palmer_b,
        "water_density_kg_m3": water_density_kg_m3,
    }

    return ds


def build_converted_dataset(
    input_zarr: Path,
    variable_name: str,
    marshall_palmer_a: float,
    marshall_palmer_b: float,
    water_density_kg_m3: float,
    output_mode: str,
) -> xr.Dataset:
    zarr_format = detect_zarr_format(input_zarr)

    ds = xr.open_zarr(
        input_zarr,
        consolidated=False,
        zarr_format=zarr_format,
        chunks={"time": 1, "y": -1, "x": -1},
    )

    if variable_name not in ds:
        raise ValueError(
            f"Variable {variable_name!r} not found. "
            f"Available variables: {list(ds.data_vars)}"
        )

    original_attrs = dict(ds[variable_name].attrs)

    if output_mode == "final-only":
        ds = build_final_only_dataset(
            ds=ds,
            variable_name=variable_name,
            original_attrs=original_attrs,
            marshall_palmer_a=marshall_palmer_a,
            marshall_palmer_b=marshall_palmer_b,
            water_density_kg_m3=water_density_kg_m3,
        )

    elif output_mode == "full":
        ds = build_full_dataset(
            ds=ds,
            variable_name=variable_name,
            original_attrs=original_attrs,
            marshall_palmer_a=marshall_palmer_a,
            marshall_palmer_b=marshall_palmer_b,
            water_density_kg_m3=water_density_kg_m3,
        )

    else:
        raise ValueError(f"Unsupported output mode: {output_mode!r}")

    ds.attrs = dict(ds.attrs)
    ds.attrs["history"] = (
        ds.attrs.get("history", "") + "\n"
        f"Converted assumed dBZ values to rainfall rate using Marshall-Palmer "
        f"relation with a={marshall_palmer_a} and b={marshall_palmer_b}. "
        f"Output mode: {output_mode}. "
        f"Final {variable_name} variable is in kg m-2 h-1."
    ).strip()

    return ds


def build_encoding(
    ds: xr.Dataset,
    zarr_format: int,
    time_shard: int,
    use_sharding: bool,
) -> dict:
    encoding = {}

    for variable_name, da in ds.data_vars.items():
        if {"time", "y", "x"}.issubset(da.dims):
            chunks = tuple(
                1 if dim == "time" else ds.sizes[dim]
                for dim in da.dims
            )

            variable_encoding = {
                "chunks": chunks,
            }

            if zarr_format == 3 and use_sharding:
                shards = tuple(
                    time_shard if dim == "time" else ds.sizes[dim]
                    for dim in da.dims
                )
                variable_encoding["shards"] = shards

            encoding[variable_name] = variable_encoding

    return encoding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert assumed dBZ radar values to rainfall rate in mm h-1 and "
            "rainfall flux in kg m-2 h-1 using the Marshall-Palmer relation."
        )
    )

    parser.add_argument("--input-zarr", type=Path, default=DEFAULT_INPUT_ZARR)
    parser.add_argument("--output-zarr", type=Path, default=DEFAULT_OUTPUT_ZARR)
    parser.add_argument("--variable-name", type=str, default=DEFAULT_VAR_NAME)

    parser.add_argument(
        "--marshall-palmer-a",
        type=float,
        default=DEFAULT_MARSHALL_PALMER_A,
    )

    parser.add_argument(
        "--marshall-palmer-b",
        type=float,
        default=DEFAULT_MARSHALL_PALMER_B,
    )

    parser.add_argument(
        "--water-density-kg-m3",
        type=float,
        default=DEFAULT_WATER_DENSITY_KG_M3,
    )

    parser.add_argument(
        "--time-shard",
        type=int,
        default=DEFAULT_TIME_SHARD,
        help="Temporal shard size for Zarr v3 variables.",
    )

    parser.add_argument(
        "--disable-sharding",
        action="store_true",
        help="Disable Zarr v3 sharding.",
    )

    parser.add_argument(
        "--output-mode",
        choices=["final-only", "full"],
        default=DEFAULT_OUTPUT_MODE,
        help=(
            "'final-only' keeps only the final variable in kg m-2 h-1. "
            "'full' also stores dBZ and mm h-1 intermediate variables."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output Zarr if it already exists.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_zarr.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output Zarr already exists: {args.output_zarr}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(args.output_zarr)

    zarr_format = detect_zarr_format(args.input_zarr)
    use_sharding = not args.disable_sharding

    ds_out = build_converted_dataset(
        input_zarr=args.input_zarr,
        variable_name=args.variable_name,
        marshall_palmer_a=args.marshall_palmer_a,
        marshall_palmer_b=args.marshall_palmer_b,
        water_density_kg_m3=args.water_density_kg_m3,
        output_mode=args.output_mode,
    )

    encoding = build_encoding(
        ds=ds_out,
        zarr_format=zarr_format,
        time_shard=args.time_shard,
        use_sharding=use_sharding,
    )

    print(f"Writing converted dataset to: {args.output_zarr}")
    print(f"Input Zarr: {args.input_zarr}")
    print(f"Input variable assumed to be dBZ: {args.variable_name}")
    print(f"Output mode: {args.output_mode}")
    print(f"Zarr format: {zarr_format}")
    print(f"Use sharding: {use_sharding and zarr_format == 3}")
    print(f"Time shard: {args.time_shard}")
    print(f"Marshall-Palmer a: {args.marshall_palmer_a}")
    print(f"Marshall-Palmer b: {args.marshall_palmer_b}")
    print(f"Water density: {args.water_density_kg_m3} kg m-3")
    print(f"Variables: {list(ds_out.data_vars)}")
    print(f"Encoding: {encoding}")

    with ProgressBar():
        ds_out.to_zarr(
            args.output_zarr,
            mode="w",
            consolidated=False,
            zarr_format=zarr_format,
            encoding=encoding,
            compute=True,
            align_chunks=True,
        )

    print("Done.")


if __name__ == "__main__":
    main()