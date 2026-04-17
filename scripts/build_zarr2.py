from pathlib import Path

from mlcast_dataset_es_aemet_reflectivity.pipeline_zarr2 import (
    RadarBuildConfig,
    run_pipeline,
)


if __name__ == "__main__":
    config = RadarBuildConfig(
        workdir=Path("./workdir_zarr2"),
        zarr_out=Path("./ES-AEMET-radar_reflectivity-ppi_ZAR_v2.zarr"),
        png_out=Path("./radar_quicklook.png"),
        png_out_cartopy=Path("./radar_quicklook_cartopy.png"),
        fechaini="20241001T000000",
        fechafin="20241002T000000",
        imagen="PPI",
        configuracion="Z_005_240",
        radar="ZAR",
        epsg="4326",
        var_name="equivalent_reflectivity_factor",
        standard_name="equivalent_reflectivity_factor",
        mlcast_created_by="Adrián García <agarciaa@aemet.es>",
        mlcast_created_with="https://github.com/mlcast-community/mlcast-dataset-ES-AEMET-reflectivity@v0.1.0",
        mlcast_dataset_version="0.1.0",
        mlcast_dataset_identifier="ES-AEMET-radar_reflectivity-ppi_ZAR",
        mlcast_dataset_identifier_format="{country_code}-{entity}-{physical_variable}-{common_name}",
        compressor_name="zstd",
        compression_level=5,
        blosc_shuffle="bitshuffle",
        zarr_format=2,
        time_chunk=1,
        use_sharding=False,
        shard_time=None,
    )
    run_pipeline(config)