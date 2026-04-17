# mlcast-dataset-ES-AEMET-reflectivity

This repository contains the code used to download AEMET radar products and convert them into an MLCAST-style Zarr dataset.

## What the pipeline does

1. Download radar files from the AEMET Big Data API in hourly windows.
2. Retry automatically when the API returns HTTP 429.
3. Standardize the NetCDF files into a common radar dataset structure.
4. Build temporary chunk-level Zarr stores.
5. Append each chunk sequentially into a final Zarr dataset to avoid excessive memory use.
6. Run quick validation and create simple diagnostic plots.

## Repository structure

```text
src/mlcast_dataset_es_aemet_reflectivity/
  data_handler.py      # API download helpers
  radar_to_zarr.py     # standardization, metadata, Zarr writing, validation and plotting
  pipeline.py          # sequential chunked build pipeline
scripts/
  build_dataset.py     # example entrypoint
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Required environment variable

The pipeline expects the AEMET API key in an environment variable:

```bash
export AEMET_API_KEY="your_api_key_here"
```

## Example execution

```bash
python scripts/build_dataset.py
```

## Notes

- The original hard-coded API key has been removed from the repository and replaced by `AEMET_API_KEY`.
- The current example is configured for the PPI reflectivity product over radar `ZAR`.
- The dataset identifier and metadata fields can be adapted for another radar, product or naming convention.

## Suggested next improvements

- Move the example configuration to a YAML or TOML file.
- Add tests for metadata and Zarr structure.
- Add CI to validate formatting and a small synthetic sample.
- Expose a CLI with `argparse` or `typer`.
