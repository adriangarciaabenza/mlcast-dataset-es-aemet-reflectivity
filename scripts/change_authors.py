from pathlib import Path

import zarr


ZARR_PATH = Path(
    "/lustre/utmp/std/MLCAST_radar_data/"
    "ES-AEMET-rainfall_rate-sri_ZAR_2020-2024_v1_converted.zarr"
)

NEW_VALUE = "Adrián García <agarciaa@aemet.es>"


def main() -> None:
    root = zarr.open_group(str(ZARR_PATH), mode="a")

    print("Old value:")
    print(root.attrs.get("mlcast_created_by"))

    root.attrs["mlcast_created_by"] = NEW_VALUE

    print("Updated value:")
    print(root.attrs.get("mlcast_created_by"))


if __name__ == "__main__":
    main()