# mlcast-dataset-ES-AEMET-reflectivity

Code to download AEMET weather-radar products and convert them into an
MLCast-compatible Zarr dataset.

The current configuration generates **PPI equivalent reflectivity** for the
single AEMET radar `ZAR` (Zaragoza). It is not a national radar composite.
The archive configured in `configs/default_v2.yaml` and
`configs/default_v3.yaml` covers 2020-01-01 through 2024-01-01.

## Pipeline

For each hourly time window, the pipeline:

1. Downloads NetCDF radar files from the AEMET Big Data API.
2. Retries requests that fail with HTTP 429 using exponential backoff.
3. Standardizes the radar variable to `(time, y, x)` and adds `lat`, `lon`
   and `spatial_ref` metadata.
4. Appends the result to a final Zarr v2 or v3 store.
5. Optionally inspects values and the generated metadata.

Missing or failed hourly windows must be checked before publishing the
resulting dataset. The output should not be considered complete without a
validation report and a record of unavailable time steps.

## Repository structure

```text
configs/
  default_v2.yaml       # Zarr v2 configuration template
  default_v3.yaml       # Zarr v3 configuration for the current archive
  tests/                # small configurations for local checks
scripts/
  build_zarr.py         # main CLI entry point
  explore_zarr.py       # inspect a generated Zarr store
  plot_zarr_gif*.py     # visualization helpers
src/mlcast_dataset_es_aemet_reflectivity/
  data_handler.py       # AEMET API download helpers
  pipeline_zarr2.py     # sequential Zarr v2 pipeline
  pipeline_zarr3.py     # sequential, sharded Zarr v3 pipeline
  radar_processing.py   # standardization and Zarr preparation
  radar_inspection.py   # validation, diagnostics and plotting
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Use a fixed environment for production builds. In particular, the Zarr v3
pipeline is developed against the Zarr 3.x API.

## AEMET credentials

An AEMET Big Data API key is required. Do not commit it to the repository or
place it in a script:

```bash
export AEMET_API_KEY="your_api_key_here"
```

The [AEMET Big Data platform](http://bigdata.aemet.es/bigdata/inicio) is for
internal access. If you are interested in reproducing the dataset preparation,
please contact Adrián García
(`agarciaa@aemet.es`) or Jaime Castro (`jcastroa@aemet.es`).

If a key has ever been committed, revoke or rotate it even if it was later
removed from the current working tree.

## Build a dataset

The main entry point accepts defaults, a YAML configuration and command-line
overrides. Command-line values take precedence over YAML values.

Example using Zarr v3:

```bash
python scripts/build_zarr.py --config configs/default_v3.yaml
```

Small local test:

```bash
python scripts/build_zarr.py --config configs/tests/small_test_v3.yaml
```

The output path, time range, radar, product, compression and MLCAST metadata
are defined in the selected YAML file. The generated Zarr stores and local
working data are intentionally excluded from Git.

## Inspect the output

```bash
python scripts/explore_zarr.py \
  --zarr-path ./ES-AEMET-radar_reflectivity-ppi_ZAR_v3.zarr \
  --var-name equivalent_reflectivity_factor
```

The main variable is expected to have:

```text
equivalent_reflectivity_factor(time, y, x)
```

Values are stored as `float32`; missing or invalid measurements are represented
by `NaN`. The dataset includes CF metadata, geographic coordinates and a CRS
in `spatial_ref`.

## Data and code licensing

The generated dataset is configured with `CC-BY-4.0` attribution metadata.
Confirm the applicable AEMET terms and access conditions before distribution.
The repository itself should include an explicit license for its source code
before publication.
