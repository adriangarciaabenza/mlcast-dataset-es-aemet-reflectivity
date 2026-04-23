from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap


# =============================================================================
# DEFAULT CONFIG
# =============================================================================

DEFAULT_ZARR_PATH = Path("./ES-AEMET-radar_reflectivity-ppi_ZAR.zarr")
DEFAULT_VAR_NAME = "equivalent_reflectivity_factor"
DEFAULT_OUTPUT_DIR = Path("./outputs")
DEFAULT_OUTPUT_GIF = DEFAULT_OUTPUT_DIR / "radar.gif"
DEFAULT_FPS = 4
DEFAULT_USE_LATLON = True


# =============================================================================
# ARGPARSE
# =============================================================================

def parse_time_or_none(value: str | None) -> str | None:
    """
    Valida que el string temporal tenga un formato interpretable por pandas/xarray.
    Devuelve el string original para que luego xarray lo use directamente en .sel().
    """
    if value is None:
        return None

    try:
        pd.to_datetime(value)
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Valor temporal no válido: {value!r}. "
            f"Usa formatos como '2020-01-01' o '2020-01-01T12:00:00'."
        ) from e

    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera un GIF animado a partir de un Zarr de radar."
    )

    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=DEFAULT_ZARR_PATH,
        help="Ruta al dataset Zarr.",
    )
    parser.add_argument(
        "--var-name",
        type=str,
        default=DEFAULT_VAR_NAME,
        help="Nombre de la variable a representar.",
    )
    parser.add_argument(
        "--output-gif",
        type=Path,
        default=DEFAULT_OUTPUT_GIF,
        help="Ruta del GIF de salida.",
    )
    parser.add_argument(
        "--start-time",
        type=parse_time_or_none,
        default=None,
        help="Tiempo inicial. Ejemplos: '2020-01-01' o '2020-01-01T12:00:00'.",
    )
    parser.add_argument(
        "--end-time",
        type=parse_time_or_none,
        default=None,
        help="Tiempo final. Ejemplos: '2020-01-01' o '2020-01-01T18:00:00'.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Frames por segundo del GIF.",
    )

    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Borra el directorio de salida antes de generar el GIF.",
    )

    latlon_group = parser.add_mutually_exclusive_group()
    latlon_group.add_argument(
        "--use-latlon",
        dest="use_latlon",
        action="store_true",
        help="Usar lon/lat en el plot.",
    )
    latlon_group.add_argument(
        "--no-use-latlon",
        dest="use_latlon",
        action="store_false",
        help="No usar lon/lat; representar en coordenadas índice.",
    )
    parser.set_defaults(use_latlon=DEFAULT_USE_LATLON)

    return parser


# =============================================================================
# ZARR HELPERS
# =============================================================================

def detect_zarr_format(path: Path) -> int:
    return 3 if (path / "zarr.json").exists() else 2


def open_radar_zarr(path: Path) -> xr.Dataset:
    zarr_format = detect_zarr_format(path)
    print(f"Abrir Zarr v{zarr_format}: {path}")

    if zarr_format == 3:
        return xr.open_zarr(path, consolidated=None, zarr_format=3)

    return xr.open_zarr(path, consolidated=True)


# =============================================================================
# OUTPUTS
# =============================================================================

def prepare_output_dir(path: Path, clean: bool = False) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# COLORMAP
# =============================================================================

def get_radar_cmap():
    """
    Colormap discreto tipo radar.

    Comportamiento:
    - valores < 12   -> negro
    - valores 12-72  -> escala discreta definida
    - valores > 72   -> #C8005A
    - NaN            -> gris
    """

    bounds = [12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78]

    colors = [
        "#0000fc",  # 12-18
        "#0094fc",  # 18-24
        "#00fcfc",  # 24-30
        "#438323",  # 30-36
        "#00c000",  # 36-42
        "#00ff00",  # 42-48
        "#ffff00",  # 48-54
        "#ffbb00",  # 54-60
        "#ff7f00",  # 60-66
        "#ff0000",  # 66-72
        "#C8005A",  # 72-78 y >78
    ]

    cmap = ListedColormap(colors)
    cmap.set_under("black")
    cmap.set_over("#C8005A")
    cmap.set_bad("#7f7f7f")

    norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=False)
    return cmap, norm, bounds


# =============================================================================
# GIF
# =============================================================================

def create_radar_gif(
    ds: xr.Dataset,
    var_name: str,
    output_gif: str | Path = "radar.gif",
    start_time: str | None = None,
    end_time: str | None = None,
    fps: int = 4,
    use_latlon: bool = False,
) -> None:
    """
    Genera un GIF animado de la variable radar con escala discreta tipo radar.
    """

    if var_name not in ds:
        raise KeyError(f"La variable {var_name!r} no existe en el dataset.")

    da = ds[var_name]

    if "time" not in da.dims:
        raise ValueError(f"La variable {var_name!r} no tiene dimensión 'time'.")

    if start_time or end_time:
        da = da.sel(time=slice(start_time, end_time))

    # Limpieza:
    # 0.0 -> NaN
    da = da.where(da != 0.0, np.nan)

    times = da.time.values

    if len(times) == 0:
        raise ValueError("No hay datos en el rango temporal seleccionado.")

    print(f"Número de frames: {len(times)}")
    print(f"Primer tiempo: {times[0]}")
    print(f"Último tiempo: {times[-1]}")

    cmap, norm, bounds = get_radar_cmap()
    frames = []

    for i, t in enumerate(times):
        print(f"Frame {i + 1}/{len(times)}", end="\r")

        data = da.sel(time=t)

        fig, ax = plt.subplots(figsize=(7, 6), facecolor="black")
        ax.set_facecolor("black")

        if use_latlon:
            if "lon" not in ds or "lat" not in ds:
                raise KeyError(
                    "Se ha pedido --use-latlon pero el dataset no contiene 'lon' y 'lat'."
                )

            mesh = ax.pcolormesh(
                ds["lon"].values,
                ds["lat"].values,
                data.values,
                shading="auto",
                cmap=cmap,
                norm=norm,
            )
            ax.set_xlabel("Lon", color="white")
            ax.set_ylabel("Lat", color="white")
        else:
            mesh = ax.imshow(
                data.values,
                origin="lower",
                cmap=cmap,
                norm=norm,
            )
            ax.set_xlabel("X", color="white")
            ax.set_ylabel("Y", color="white")

        ax.set_title(str(t), color="white")
        ax.tick_params(colors="white")

        for spine in ax.spines.values():
            spine.set_color("white")

        cbar = plt.colorbar(
            mesh,
            ax=ax,
            orientation="horizontal",
            pad=0.08,
            fraction=0.06,
            boundaries=bounds,
            ticks=bounds[:-1],
            spacing="uniform",
            extend="max",
        )

        cbar.set_label("Reflectividad (dBZ)", color="white")
        cbar.ax.xaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.get_xticklabels(), color="white")
        cbar.outline.set_edgecolor("white")

        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())
        frames.append(image)

        plt.close(fig)

    print("\nFrames generados.")

    output_gif = Path(output_gif)
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_gif, frames, fps=fps)

    print(f"GIF guardado en: {output_gif}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    prepare_output_dir(args.output_gif.parent, clean=args.clean_output)

    ds = open_radar_zarr(args.zarr_path)
    print(ds)

    create_radar_gif(
        ds=ds,
        var_name=args.var_name,
        output_gif=args.output_gif,
        start_time=args.start_time,
        end_time=args.end_time,
        fps=args.fps,
        use_latlon=args.use_latlon,
    )


if __name__ == "__main__":
    main()