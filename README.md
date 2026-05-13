# Hydro Image Discharge Pipeline

This repository provides a Python workflow for estimating river discharge from
fixed-camera imagery. It downloads camera images and USGS discharge records,
matches image timestamps to discharge observations, filters low-quality images,
prepares flow-state ROI variants, extracts ConvNeXt-Tiny image embeddings,
trains a mixture-of-experts discharge model, and compares the model against
Random Forest and SVM baselines.

The repository contains source code, configuration templates, and method
documentation only. Downloaded images, discharge tables, ROI files, extracted
features, model outputs, and logs are intentionally excluded from git.

## Install

Create or activate a Python environment with the required dependencies:

```text
numpy
pandas
requests
opencv-python
Pillow
scikit-learn
torch
torchvision
```

Install the package in editable mode:

```bash
python -m pip install -e .
```

If the dependencies are already installed in your environment:

```bash
python -m pip install -e . --no-deps
```

Available command-line entry points:

```bash
hydro-prepare-data --help
hydro-filter-images --help
hydro-select-roi-references --help
hydro-run-roi-variants --help
hydro-run-sklearn-baselines --help
```

These commands are wrappers around the Python scripts in `hydro_image_pipeline`.
They do not skip any step of the workflow.

## Repository Layout

```text
hydro_image_pipeline/
  configs/
    best_pipeline_config.json
    site_list.example.csv
  get_data/
    prepare_data.py
    filter_images.py
    filter_site_images.py
    select_roi_reference_images.py
  train/
    extract_torchvision_backbone_features_v7.py
    prepare_site_dataset.py
    run_best_pipeline.py
    run_newroi_variants.py
    run_newroi_sklearn_baselines.py
README.md
README_GITHUB.md
TECHNICAL_METHODS.md
pyproject.toml
```

Generated folders are ignored by git:

```text
data/
results/
logs/
```

## 1. Prepare a Site List

Copy the example file:

```bash
cp hydro_image_pipeline/configs/site_list.example.csv \
  hydro_image_pipeline/configs/site_list.csv
```

Edit `hydro_image_pipeline/configs/site_list.csv`.

Required columns:

```csv
river_group,site_folder,cam_id,nwis_id
```

Example:

```csv
river_group,site_folder,cam_id,nwis_id
Example_River,Example_Camera_Name,Example_Camera_ID,01234567
```

Use one row per camera. Cameras observing the same river reach can share the
same `river_group`. Cameras using the same USGS gage should share the same
`nwis_id`, so the discharge record is reused. Only include sites with USGS
discharge parameter `00060`.

## 2. Download Images and Discharge

```bash
hydro-prepare-data \
  --site-csv hydro_image_pipeline/configs/site_list.csv \
  --output-root data \
  --limit 50000 \
  --hydro-buffer-days 1 \
  --max-gap-minutes 90
```

Expected outputs:

```text
data/{river_group}/{camera_name}/images_all/
data/{river_group}/{camera_name}/matched_all.csv
data/{river_group}/{camera_name}/matched.csv
data/{river_group}/discharge/{nwis_id}/long_table.csv
```

`matched_all.csv` records all attempted image-discharge matches. `matched.csv`
contains the valid matches used by downstream stages.

## 3. Filter Images and Create Splits

```bash
hydro-filter-images \
  --output-root data \
  --test-water-year 2025 \
  --fallback-val-ratio 0.2 \
  --rough-year-min-months 10 \
  --seed 42 \
  --day-start 8.0 \
  --day-end 18.5 \
  --activate-filtered
```

This removes low-quality images such as night, blur, low contrast, fog,
wet-lens artifacts, glare, snow, and ice.

Expected outputs:

```text
data/{river_group}/{camera_name}/train/
data/{river_group}/{camera_name}/val/
data/{river_group}/{camera_name}/test/
data/{river_group}/{camera_name}/images_filtered_out/
data/{river_group}/{camera_name}/image_quality_audit.csv
data/{river_group}/{camera_name}/split_manifest.csv
data/{river_group}/{camera_name}/filter_summary.json
```

The default test set is water year 2025. Validation is chosen from pre-test data
using roughly full-year blocks when enough data exist; otherwise it is sampled
seasonally.

## 4. Select ROI Reference Images

After filtering, select one clean training image near each target discharge
percentile:

```bash
hydro-select-roi-references \
  --data-root data \
  --output-root data/roi_variants \
  --overwrite
```

For each camera, this creates:

```text
data/roi_variants/{camera_name}/reference_images/
data/roi_variants/{camera_name}/roi_reference_manifest.csv
```

The reference images correspond to:

```text
p05: closest clean training image to the 5th percentile discharge
p50: closest clean training image to the 50th percentile discharge
p95: closest clean training image to the 95th percentile discharge
```

Draw one ROI polygon on each reference image and save:

```text
data/roi_variants/{camera_name}/{camera_name}_p05.json
data/roi_variants/{camera_name}/{camera_name}_p50.json
data/roi_variants/{camera_name}/{camera_name}_p95.json
```

Each ROI should follow the visible water boundary for that flow condition and
exclude obvious non-water areas when possible.

## 5. Create the Modeling Site List

Create `data/pipeline_site_list.csv` after the site has been filtered and the
ROI JSON files are ready.

Format:

```csv
site_id,site_dir
Example_Camera_Name,data/Example_River/Example_Camera_Name
```

This file is an allowlist for the modeling stage. You can download many cameras
but train only the cameras that are ready for the current experiment.

## 6. Run MoE Models for p05, p50, and p95 ROIs

List planned tasks:

```bash
hydro-run-roi-variants \
  --site-list-csv data/pipeline_site_list.csv \
  --newroi-root data/roi_variants \
  --output-root results/moe_roi_variants \
  --work-root results/moe_roi_variants/_inputs \
  --config hydro_image_pipeline/configs/best_pipeline_config.json \
  --list-tasks
```

Run a task:

```bash
hydro-run-roi-variants \
  --site-list-csv data/pipeline_site_list.csv \
  --newroi-root data/roi_variants \
  --output-root results/moe_roi_variants \
  --work-root results/moe_roi_variants/_inputs \
  --config hydro_image_pipeline/configs/best_pipeline_config.json \
  --task-index 0
```

Expected outputs:

```text
results/moe_roi_variants/p05/{camera_name}/features_convnext_tiny.csv
results/moe_roi_variants/p05/{camera_name}/metrics.csv
results/moe_roi_variants/p05/{camera_name}/predictions.csv

results/moe_roi_variants/p50/{camera_name}/...
results/moe_roi_variants/p95/{camera_name}/...
```

The feature extraction stage applies the ROI, optionally removes likely
vegetation/bank pixels inside the ROI, normalizes contrast, and extracts
ConvNeXt-Tiny embeddings.

## 7. Run Random Forest and SVM Baselines

Run after `features_convnext_tiny.csv` files exist from the MoE stage:

```bash
hydro-run-sklearn-baselines \
  --feature-root results/moe_roi_variants \
  --output-root results/sklearn_baselines
```

The baseline stage scans all available p05, p50, and p95 embedding files. It
does not use only the best MoE ROI. RF and SVM are evaluated separately for each
ROI variant so the comparison is fair.

Expected outputs:

```text
results/sklearn_baselines/p05/{camera_name}/metrics.csv
results/sklearn_baselines/p50/{camera_name}/metrics.csv
results/sklearn_baselines/p95/{camera_name}/metrics.csv
```

## 8. Review Results

For each camera, compare:

```text
MoE p05 vs MoE p50 vs MoE p95
RF p05 vs RF p50 vs RF p95
SVM p05 vs SVM p50 vs SVM p95
```

Then compare the best ROI result from each model family:

```text
best MoE vs best RF vs best SVM
```

Primary metrics:

```text
R2
RMSE
MAE
```

For the MoE model, the soft prediction is generally the main result because it
uses a continuous learned gate to combine low-flow and high-flow experts.

## Running on a Server or Cluster

This repository does not include cluster submission scripts. If you run the
workflow on a server, HPC cluster, or cloud instance, submit the commands above
using that system's own job scheduler, resource account, queue, GPU request, and
storage paths.

## Method Details

See [TECHNICAL_METHODS.md](TECHNICAL_METHODS.md) for the methods-style
description of image filtering thresholds, ROI selection, water-mask cleanup,
ConvNeXt feature extraction, the mixture-of-experts model, and baseline
comparisons.
