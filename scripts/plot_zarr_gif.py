from __future__ import annotations

import shutil
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap


# =============================================================================
# CONFIG
# =============================================================================

ZARR_PATH = Path("./ES-AEMET-radar_reflectivity-ppi_ZAR.zarr")
VAR_NAME = "equivalent_reflectivity_factor"

OUTPUT_DIR = Path("./outputs")
OUTPUT_GIF = OUTPUT_DIR / "radar.gif"

START_TIME = None
END_TIME = None

FPS = 4
USE_LATLON = True


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

def prepare_output_dir(path: Path) -> None:
    if path.exists():
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
    cmap.set_under("black")      # <12
    cmap.set_over("#C8005A")     # >78
    cmap.set_bad("#7f7f7f")      # NaN

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

    da = ds[var_name]

    if start_time or end_time:
        da = da.sel(time=slice(start_time, end_time))

    # Limpieza igual que en tu notebook:
    # 0.0 -> NaN
    da = da.where(da != 0.0, np.nan)

    times = da.time.values

    if len(times) == 0:
        raise ValueError("No hay datos en el rango temporal seleccionado.")

    print(f"Número de frames: {len(times)}")

    cmap, norm, bounds = get_radar_cmap()

    frames = []

    for i, t in enumerate(times):
        print(f"Frame {i + 1}/{len(times)}", end="\r")

        data = da.sel(time=t)

        fig, ax = plt.subplots(figsize=(7, 6), facecolor="black")
        ax.set_facecolor("black")

        if use_latlon:
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
    prepare_output_dir(OUTPUT_DIR)

    ds = open_radar_zarr(ZARR_PATH)
    print(ds)

    create_radar_gif(
        ds=ds,
        var_name=VAR_NAME,
        output_gif=OUTPUT_GIF,
        start_time=START_TIME,
        end_time=END_TIME,
        fps=FPS,
        use_latlon=USE_LATLON,
    )


if __name__ == "__main__":
    main()