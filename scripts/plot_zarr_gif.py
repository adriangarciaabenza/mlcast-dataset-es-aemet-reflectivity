from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature


# =============================================================================
# DEFAULT CONFIG
# =============================================================================

DEFAULT_ZARR_PATH = Path("./ES-AEMET-radar_reflectivity-ppi_ZAR.zarr")
DEFAULT_VAR_NAME = "equivalent_reflectivity_factor"
DEFAULT_OUTPUT_GIF = Path("./outputs/radar.gif")
DEFAULT_FPS = 4
DEFAULT_USE_LATLON = True


# =============================================================================
# ARGPARSE
# =============================================================================

def parse_time_or_none(value: str | None) -> str | None:
    if value is None:
        return None

    try:
        pd.to_datetime(value)
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Invalid time value: {value!r}. "
            f"Use formats such as '2020-01-01' or '2020-01-01T12:00:00'."
        ) from e

    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an animated GIF from a radar Zarr dataset."
    )

    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=DEFAULT_ZARR_PATH,
        help="Path to the Zarr dataset.",
    )
    parser.add_argument(
        "--var-name",
        type=str,
        default=DEFAULT_VAR_NAME,
        help="Name of the variable to display.",
    )
    parser.add_argument(
        "--output-gif",
        type=Path,
        default=DEFAULT_OUTPUT_GIF,
        help="Path to the output GIF.",
    )
    parser.add_argument(
        "--start-time",
        type=parse_time_or_none,
        default=None,
        help="Start time. Examples: '2020-01-01' or '2020-01-01T12:00:00'.",
    )
    parser.add_argument(
        "--end-time",
        type=parse_time_or_none,
        default=None,
        help="End time. Examples: '2020-01-01' or '2020-01-01T18:00:00'.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="GIF frames per second.",
    )

    latlon_group = parser.add_mutually_exclusive_group()
    latlon_group.add_argument(
        "--use-latlon",
        dest="use_latlon",
        action="store_true",
        help="Use lon/lat in the basic plot.",
    )
    latlon_group.add_argument(
        "--no-use-latlon",
        dest="use_latlon",
        action="store_false",
        help="Do not use lon/lat; plot in index coordinates instead.",
    )
    parser.set_defaults(use_latlon=DEFAULT_USE_LATLON)

    parser.add_argument(
        "--use-cartopy",
        action="store_true",
        help="Plot with Cartopy.",
    )

    # -------------------------------------------------------------------------
    # Administrative lines / provinces
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--draw-provinces",
        action="store_true",
        help=(
            "Overlay internal administrative lines from Natural Earth "
            "(via Cartopy)."
        ),
    )
    parser.add_argument(
        "--draw-shapefile",
        action="store_true",
        help="Overlay any external shapefile.",
    )
    parser.add_argument(
        "--shapefile-path",
        type=Path,
        default=None,
        help="Path to the external shapefile (.shp).",
    )
    parser.add_argument(
        "--shapefile-edgecolor",
        type=str,
        default="white",
        help="Edge color for the external shapefile.",
    )
    parser.add_argument(
        "--shapefile-linewidth",
        type=float,
        default=0.6,
        help="Line width for the external shapefile.",
    )

    # -------------------------------------------------------------------------
    # Orography
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--draw-orography",
        action="store_true",
        help="Overlay orography.",
    )
    parser.add_argument(
        "--orography-mode",
        type=str,
        choices=["default", "file", "shapefile"],
        default="default",
        help=(
            "Orography mode: "
            "'default' = Cartopy default physical layer, "
            "'file' = external raster/NetCDF, "
            "'shapefile' = external shapefile."
        ),
    )

    parser.add_argument(
        "--orography-file",
        type=Path,
        default=None,
        help="Path to a NetCDF/raster file with orography.",
    )
    parser.add_argument(
        "--orography-var",
        type=str,
        default="orography",
        help="Name of the orography variable in the external file.",
    )

    parser.add_argument(
        "--orography-shapefile",
        type=Path,
        default=None,
        help="Path to an external shapefile for orography/relief.",
    )
    parser.add_argument(
        "--orography-edgecolor",
        type=str,
        default="white",
        help="Edge color for orography in shapefile mode.",
    )
    parser.add_argument(
        "--orography-linewidth",
        type=float,
        default=0.5,
        help="Line width for orography in shapefile mode.",
    )

    return parser


# =============================================================================
# ZARR HELPERS
# =============================================================================

def detect_zarr_format(path: Path) -> int:
    return 3 if (path / "zarr.json").exists() else 2


def open_radar_zarr(path: Path) -> xr.Dataset:
    zarr_format = detect_zarr_format(path)
    print(f"Opening Zarr v{zarr_format}: {path}")

    if zarr_format == 3:
        return xr.open_zarr(path, consolidated=None, zarr_format=3)

    return xr.open_zarr(path, consolidated=True)


# =============================================================================
# COLORMAP
# =============================================================================

def get_radar_cmap():
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
        "#C8005A",  # 72-78 and >78
    ]

    cmap = ListedColormap(colors)
    cmap.set_under("black")
    cmap.set_over("#C8005A")
    cmap.set_bad("#7f7f7f")

    norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=False)
    return cmap, norm, bounds


# =============================================================================
# OPTIONAL OVERLAYS
# =============================================================================

def add_default_admin1_lines(ax, scale: str = "10m") -> None:
    """
    Default internal administrative lines from Natural Earth/Cartopy.
    """
    feature = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale=scale,
        facecolor="none",
    )
    ax.add_feature(
        feature,
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )


def add_shapefile(
    ax,
    shapefile_path: Path | None,
    edgecolor: str = "white",
    linewidth: float = 0.5,
    zorder: int = 4,
) -> None:
    if shapefile_path is None:
        print("Warning: a shapefile was requested but no path was provided.")
        return

    if not shapefile_path.exists():
        print(f"Warning: shapefile does not exist: {shapefile_path}")
        return

    feature = ShapelyFeature(
        Reader(str(shapefile_path)).geometries(),
        ccrs.PlateCarree(),
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_feature(feature, zorder=zorder)


def add_default_orography(ax, scale: str = "10m") -> None:
    """
    Add a default general physical relief layer from Natural Earth.
    This is not a detailed raster orography; it is a vector physical reference.
    """
    try:
        feature = cfeature.NaturalEarthFeature(
            category="physical",
            name="geography_regions_elevation_points",
            scale=scale,
            facecolor="none",
        )
        ax.add_feature(
            feature,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.5,
            zorder=3,
        )
    except Exception as e:
        print(f"Warning: could not add default orography: {e}")


def add_orography_from_file(
    ax,
    orography_file: Path | None,
    var_name: str = "orography",
) -> None:
    if orography_file is None:
        print("Warning: file-based orography was requested but no path was provided.")
        return

    if not orography_file.exists():
        print(f"Warning: orography file does not exist: {orography_file}")
        return

    ds_oro = xr.open_dataset(orography_file)

    if var_name not in ds_oro:
        ds_oro.close()
        raise KeyError(
            f"Orography variable {var_name!r} does not exist in {orography_file}"
        )

    oro = ds_oro[var_name]

    lon_name = "lon" if "lon" in oro.coords else "longitude"
    lat_name = "lat" if "lat" in oro.coords else "latitude"

    if lon_name not in oro.coords or lat_name not in oro.coords:
        ds_oro.close()
        raise KeyError(
            "lon/lat or longitude/latitude coordinates were not found "
            "in the orography file."
        )

    ax.contour(
        oro[lon_name].values,
        oro[lat_name].values,
        oro.values,
        levels=[500, 1000, 1500, 2000],
        colors="white",
        linewidths=0.4,
        alpha=0.5,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )

    ds_oro.close()


def add_orography(
    ax,
    mode: str = "default",
    orography_file: Path | None = None,
    orography_var: str = "orography",
    orography_shapefile: Path | None = None,
    edgecolor: str = "white",
    linewidth: float = 0.5,
) -> None:
    if mode == "default":
        add_default_orography(ax)
    elif mode == "file":
        add_orography_from_file(
            ax=ax,
            orography_file=orography_file,
            var_name=orography_var,
        )
    elif mode == "shapefile":
        add_shapefile(
            ax=ax,
            shapefile_path=orography_shapefile,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=3,
        )
    else:
        raise ValueError(f"Unknown orography mode: {mode}")


# =============================================================================
# FRAME DRAWING
# =============================================================================

def plot_frame_cartopy(
    ds: xr.Dataset,
    data: xr.DataArray,
    t,
    cmap,
    norm,
    bounds,
    draw_provinces: bool = False,
    draw_shapefile: bool = False,
    shapefile_path: Path | None = None,
    shapefile_edgecolor: str = "white",
    shapefile_linewidth: float = 0.6,
    draw_orography: bool = False,
    orography_mode: str = "default",
    orography_file: Path | None = None,
    orography_var: str = "orography",
    orography_shapefile: Path | None = None,
    orography_edgecolor: str = "white",
    orography_linewidth: float = 0.5,
):
    fig = plt.figure(figsize=(7, 6), facecolor="black")
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_facecolor("black")

    if "lon" not in ds or "lat" not in ds:
        raise KeyError(
            "Using Cartopy requires 'lon' and 'lat' coordinates in the dataset."
        )

    mesh = ax.pcolormesh(
        ds["lon"].values,
        ds["lat"].values,
        data.values,
        shading="auto",
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    lon_vals = ds["lon"].values
    lat_vals = ds["lat"].values
    ax.set_extent(
        [
            float(np.nanmin(lon_vals)),
            float(np.nanmax(lon_vals)),
            float(np.nanmin(lat_vals)),
            float(np.nanmax(lat_vals)),
        ],
        crs=ccrs.PlateCarree(),
    )

    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.5, zorder=5)

    if draw_provinces:
        add_default_admin1_lines(ax, scale="10m")

    if draw_shapefile:
        add_shapefile(
            ax=ax,
            shapefile_path=shapefile_path,
            edgecolor=shapefile_edgecolor,
            linewidth=shapefile_linewidth,
            zorder=4,
        )

    if draw_orography:
        add_orography(
            ax=ax,
            mode=orography_mode,
            orography_file=orography_file,
            orography_var=orography_var,
            orography_shapefile=orography_shapefile,
            edgecolor=orography_edgecolor,
            linewidth=orography_linewidth,
        )

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color="white",
        alpha=0.3,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(str(t), color="white")

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

    cbar.set_label("Reflectivity (dBZ)", color="white")
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_xticklabels(), color="white")
    cbar.outline.set_edgecolor("white")

    return fig


def plot_frame_basic(
    ds: xr.Dataset,
    data: xr.DataArray,
    t,
    cmap,
    norm,
    bounds,
    use_latlon: bool = False,
):
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="black")
    ax.set_facecolor("black")

    if use_latlon:
        if "lon" not in ds or "lat" not in ds:
            raise KeyError(
                "lon/lat plotting was requested, but the dataset does not contain 'lon' and 'lat'."
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

    cbar.set_label("Reflectivity (dBZ)", color="white")
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_xticklabels(), color="white")
    cbar.outline.set_edgecolor("white")

    return fig


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
    use_cartopy: bool = False,
    draw_provinces: bool = False,
    draw_shapefile: bool = False,
    shapefile_path: Path | None = None,
    shapefile_edgecolor: str = "white",
    shapefile_linewidth: float = 0.6,
    draw_orography: bool = False,
    orography_mode: str = "default",
    orography_file: Path | None = None,
    orography_var: str = "orography",
    orography_shapefile: Path | None = None,
    orography_edgecolor: str = "white",
    orography_linewidth: float = 0.5,
) -> None:
    if var_name not in ds:
        raise KeyError(f"Variable {var_name!r} does not exist in the dataset.")

    da = ds[var_name]

    if "time" not in da.dims:
        raise ValueError(f"Variable {var_name!r} does not have a 'time' dimension.")

    if start_time or end_time:
        da = da.sel(time=slice(start_time, end_time))

    # 0.0 -> NaN
    da = da.where(da != 0.0, np.nan)

    times = da.time.values
    if len(times) == 0:
        raise ValueError("There are no data in the selected time range.")

    print(f"Number of frames: {len(times)}")
    print(f"First time: {times[0]}")
    print(f"Last time: {times[-1]}")

    cmap, norm, bounds = get_radar_cmap()
    frames = []

    for i, t in enumerate(times):
        print(f"Frame {i + 1}/{len(times)}", end="\r")
        data = da.sel(time=t)

        if use_cartopy:
            fig = plot_frame_cartopy(
                ds=ds,
                data=data,
                t=t,
                cmap=cmap,
                norm=norm,
                bounds=bounds,
                draw_provinces=draw_provinces,
                draw_shapefile=draw_shapefile,
                shapefile_path=shapefile_path,
                shapefile_edgecolor=shapefile_edgecolor,
                shapefile_linewidth=shapefile_linewidth,
                draw_orography=draw_orography,
                orography_mode=orography_mode,
                orography_file=orography_file,
                orography_var=orography_var,
                orography_shapefile=orography_shapefile,
                orography_edgecolor=orography_edgecolor,
                orography_linewidth=orography_linewidth,
            )
        else:
            fig = plot_frame_basic(
                ds=ds,
                data=data,
                t=t,
                cmap=cmap,
                norm=norm,
                bounds=bounds,
                use_latlon=use_latlon,
            )

        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())
        frames.append(image)
        plt.close(fig)

    print("\nFrames generated.")

    output_gif = Path(output_gif)
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_gif, frames, fps=fps)

    print(f"GIF saved to: {output_gif}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.output_gif.parent.mkdir(parents=True, exist_ok=True)

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
        use_cartopy=args.use_cartopy,
        draw_provinces=args.draw_provinces,
        draw_shapefile=args.draw_shapefile,
        shapefile_path=args.shapefile_path,
        shapefile_edgecolor=args.shapefile_edgecolor,
        shapefile_linewidth=args.shapefile_linewidth,
        draw_orography=args.draw_orography,
        orography_mode=args.orography_mode,
        orography_file=args.orography_file,
        orography_var=args.orography_var,
        orography_shapefile=args.orography_shapefile,
        orography_edgecolor=args.orography_edgecolor,
        orography_linewidth=args.orography_linewidth,
    )


if __name__ == "__main__":
    main()