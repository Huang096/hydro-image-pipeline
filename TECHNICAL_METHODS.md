# Technical Methods for the Hydro Image Discharge Pipeline

This document describes the technical design of the image-based discharge
prediction pipeline. It is written as a methods-style explanation for readers
with hydrology expertise who may not be familiar with the implementation
details of computer vision and machine learning workflows.

## 1. Data Acquisition and Image-Discharge Matching

For each camera site, the pipeline downloads all available camera images and
the corresponding USGS discharge record. Images are first stored without quality
filtering, so that the filtering decision is reproducible and auditable.

Each image timestamp is matched to the nearest available discharge observation
from the associated USGS gage. A match is retained only if the discharge record
is close enough in time to the image timestamp. The current matching tolerance is
90 minutes. This tolerance is intended to handle small differences between image
capture time and hydrologic record time while avoiding matches that are too far
apart to represent the same hydraulic condition.

When multiple cameras observe the same river reach and use the same USGS gage,
the discharge record is downloaded once at the river-group level and reused for
all cameras in that group. This avoids duplicate hydrologic data downloads and
ensures that cameras from the same river are aligned to the same discharge time
series.

Main outputs from this stage include:

```text
images_all/
matched_all.csv
matched.csv
discharge/{nwis_id}/long_table.csv
```

`matched_all.csv` contains all attempted image-discharge matches. `matched.csv`
contains the valid matches used by downstream steps.

## 2. Image Quality Filtering

The image quality filter removes images that are unlikely to contain usable
visual information about open-channel water conditions. This step is important
because the model is trained to infer discharge from visible water-surface and
channel-state cues. Images dominated by darkness, snow, ice, glare, fog, or
lens artifacts can introduce misleading visual patterns and degrade model
performance.

The filter uses a combination of time-of-day rules, image brightness and
contrast statistics, blur metrics, color-space thresholds, and object-like blob
features. Each image receives a set of quality flags and a final `quality_keep`
decision.

### 2.1 Daytime and Night Filtering

The pipeline excludes nighttime images using both timestamp information and
image-level appearance.

Timestamp-based filtering uses local camera time and keeps only images within
the configured daytime window. The default daytime window is approximately
08:00 to 18:00 local time, with site-specific timezone handling.

Image-based nighttime filtering is also applied because timestamps alone may
not capture dark weather conditions, shadows, or poor illumination. Images are
flagged as night or near-night when they are nearly grayscale or when the image
has both low mean brightness and low peak brightness. The default dark-night
criteria are:

```text
gray_mean < 75
brightness_peak_v < 0.42
```

The rationale is that true water-surface texture, bank geometry, and flow
features are generally not visible in very dark images, even if the timestamp is
nominally during daytime.

### 2.2 Dark, Low-Contrast, and Blurry Images

The filter uses adaptive thresholds for darkness, contrast, and blur. Rather
than using only fixed thresholds, it computes site-specific thresholds from the
distribution of readable, non-night images.

The adaptive darkness threshold is:

```text
max(85, 0.90 * site 5th percentile of gray_mean)
```

The adaptive low-contrast threshold is:

```text
max(26, 0.75 * site 5th percentile of gray_std)
```

The adaptive blur threshold is based on the variance of the Laplacian, a common
focus/sharpness metric:

```text
max(250, 0.65 * site 10th percentile of lap_var)
```

Images are also dropped under more severe fixed criteria:

```text
severe_dark: gray_mean < 55
severe_low_contrast: gray_std < 18
severe_blur: lap_var < 160
```

These thresholds serve two purposes. The fixed lower bounds prevent extremely
poor images from being retained. The site-adaptive component accounts for
camera-to-camera differences in exposure, viewpoint, lens quality, and local
lighting.

### 2.3 Rain, Wet Lens, Fog, and Obstruction

The filter identifies rain and wet-lens artifacts using bright, low-saturation
blob features. These artifacts often appear as translucent or bright droplets
that distort the water surface and bank edges.

The current wet-lens/rain logic uses:

```text
raindrop_min_blob_count = 6
raindrop_min_blob_area_ratio = 0.0008
raindrop_max_blob_area_ratio = 0.06
raindrop_min_bright_low_sat_ratio = 0.0015
rain_peak_v_threshold = 0.60
```

Fog or water vapor is flagged when the image is moderately bright but has low
contrast and low colorfulness:

```text
gray_mean >= 60
gray_std < adaptive fog contrast threshold
colorfulness < 14
```

Obscured or washed-out images are detected using a combination of high
bright-low-saturation area, low contrast, and low colorfulness:

```text
bright_low_sat_ratio >= 0.10
gray_std < 42
colorfulness < 18
```

These rules are designed to remove images where the camera is technically
daytime and readable but the river surface or banks are visually unreliable.

### 2.4 Strong Glare and Overexposure

Strong glare is common in river-camera images because water surfaces can reflect
direct sunlight. The filter flags glare using bright, low-saturation pixels and
saturated bright pixels:

```text
glare_v_thresh = 0.96
glare_s_thresh = 0.18
glare_very_bright_low_sat_ratio >= 0.16
glare_saturated_ratio >= 0.15
glare_bright_low_sat_ratio >= 0.16
brightness_peak_v >= 0.88
```

The purpose is not to remove every reflection. Small reflections may still
contain useful hydrologic information. The filter targets images where glare is
large enough to dominate the ROI and obscure water texture or channel
boundaries.

### 2.5 Snow and Ice Filtering

Snow and ice are treated as low-quality conditions for this discharge-from-image
model because the visible surface may no longer represent open-channel water.
For example, a camera image of a snow-covered or ice-covered channel can have a
very different visual-discharge relationship from an open-water image.

Snow and ice checks are active during winter and shoulder-season months:

```text
snow_months = November, December, January, February, March, April
```

The snow/ice detector combines several color and texture criteria:

```text
white low-saturation snow:
  V > 0.72, S < 0.28, white_low_sat_ratio >= 0.30

large loose snow:
  white_low_sat_ratio >= 0.25
  gray_mean >= 105
  colorfulness <= 28

strong snow:
  white_low_sat_ratio >= 0.50
  gray_mean >= 105
  colorfulness <= 28

bright loose snow:
  V > 0.45, S < 0.55, bright_loose_ratio >= 0.65
  gray_mean >= 90
  gray_std <= 35

cool bright snow:
  V > 0.45, S < 0.55, cool_bright_ratio >= 0.55
  gray_mean >= 95
  gray_std <= 42

blue-dominant snow/ice:
  V > 0.35, S > 0.30
  blue channel exceeds red by at least 0.12
  blue_dominant_ratio >= 0.85

gray ice:
  V > 0.45, S < 0.18, gray_ice_low_sat_ratio >= 0.55
  gray_mean >= 95
  gray_std <= 38
  colorfulness <= 35
```

The multiple snow/ice rules are intentional. Snow and ice can appear white,
gray, blue, or low-saturation depending on lighting, camera exposure, and
surface condition. A single RGB threshold is usually too brittle for this task.

### 2.6 Filter Outputs and Auditability

For each camera, the filter produces:

```text
image_quality_audit.csv
matched_filtered.csv
water_year_summary_filtered.csv
split_manifest_filtered.csv
filter_summary.json
```

`image_quality_audit.csv` records the quality flags and reasons for each image.
This is important for scientific reproducibility: images are not silently
removed; the reason for exclusion is retained.

When the filtered results are activated, the filtered files replace the active
model inputs:

```text
matched.csv
split_manifest.csv
water_year_summary.csv
```

Rejected images are stored separately in:

```text
images_filtered_out/
```

## 3. Train/Validation/Test Splitting

Splitting is performed after quality filtering. This means that the model is
trained and evaluated only on images that passed the quality screen.

The default test period is water year 2025. Pre-test data are used for training
and validation. If enough pre-test data exist, the pipeline uses roughly full
water-year blocks: earlier complete water years for training and the most recent
pre-test water year for validation. If there is not enough full-year coverage,
the validation set is sampled seasonally from the pre-test data.(20%)

The purpose is to preserve a temporally meaningful test set while still
ensuring that the validation set contains seasonal variation.

## 4. ROI Selection Strategy

The pipeline evaluates three ROI definitions per camera. These ROIs correspond
to low-flow, median-flow, and high-flow conditions:

```text
p05: image closest to the 5th percentile of training-set discharge
p50: image closest to the 50th percentile of training-set discharge
p95: image closest to the 95th percentile of training-set discharge
```

Importantly, these percentiles are computed from the clean training images, not
from the test set. This avoids using test information during ROI selection.

The selected reference images are copied to:

```text
pipeline/data/roi_variants/{camera_name}/reference_images/
```

Then draws one ROI on each reference image. The ROI should follow the
best visual estimate of the water boundary for that flow condition. In practice,
this means drawing the polygon along the visible edge of the water surface and
excluding obvious non-water regions such as banks, vegetation, bridges, sky, and
nearby infrastructure whenever possible.

The three-ROI design is motivated by the fact that the visible water area can
change substantially with discharge. A low-flow ROI may be too narrow for
high-flow conditions, while a high-flow ROI may include exposed banks or
vegetation during low-flow conditions. Testing p05, p50, and p95 ROIs allows us
to quantify whether model performance is sensitive to the chosen flow-state ROI.

ROI files are saved as:

```text
pipeline/data/roi_variants/{camera_name}/{camera_name}_p05.json
pipeline/data/roi_variants/{camera_name}/{camera_name}_p50.json
pipeline/data/roi_variants/{camera_name}/{camera_name}_p95.json
```

## 5. ROI Water-Mask Refinement

After the hand-drawn ROI is applied, the pipeline performs an additional
computer-vision refinement step to remove high-confidence non-water pixels
inside the ROI. This step is enabled by default.

The goal is not to replace manual ROI drawing. Instead, it provides a secondary
cleanup pass for small amounts of vegetation, grass, or riverbank that remain
inside the polygon.

The refinement uses HSV and RGB color rules to detect:

```text
green vegetation
bright grass
warm/red bank material near the ROI edge
yellow bank material near the ROI edge
```

Vegetation-like pixels are identified by hue, saturation, brightness, and excess
green:

```text
hue between 34 and 92
saturation >= 62
value >= 45
excess_green >= 34
green >= 1.10 * red
green >= 1.00 * blue
```

Bank-like pixels are only removed near the edge of the hand-drawn ROI. The edge
band is:

```text
max(8 pixels, 4.5% of the smaller image dimension)
```

This edge restriction reduces the risk of removing true water pixels in the
center of the channel. Detected non-water components are morphologically opened
and dilated, then filtered by connected-component area. Very tiny components
and extremely large components are ignored.

The final refined mask is accepted only if at least 40% of the original ROI
area remains. If the refinement would remove too much of the ROI, the pipeline
falls back to the original hand-drawn ROI. This conservative fallback prevents
over-aggressive masking from destroying the water signal.

After masking, pixels outside the refined ROI are set to zero, and the image is
cropped to the ROI bounding box. CLAHE contrast normalization is applied to the
cropped ROI before feature extraction.

## 6. Image Feature Extraction

The model uses a pretrained ConvNeXt-Tiny backbone to extract visual embeddings
from each ROI image. ConvNeXt-Tiny is initialized with ImageNet pretrained
weights and used as a feature extractor.

For each image:

1. the ROI polygon is loaded,
2. the ROI is scaled to the current image size,
3. the optional water-mask refinement removes likely vegetation and bank pixels,
4. the ROI crop is contrast-normalized,
5. ConvNeXt-Tiny extracts a fixed-length embedding vector.

The output is:

```text
features_convnext_tiny.csv
```

This file contains image metadata, discharge labels, split labels, and embedding
columns named `emb_000`, `emb_001`, etc.

The same extraction process is repeated separately for p05, p50, and p95 ROI
definitions. This produces separate embeddings for each ROI condition.

## 7. Mixture-of-Experts Discharge Model

The main prediction model is a neural mixture-of-experts model designed to
handle the fact that visual-discharge relationships may differ between lower
and higher flow conditions.

The model has:

```text
one low-flow expert
one high-flow expert
one learned gate
```

The low-flow and high-flow experts each predict a non-negative discharge value.
The gate predicts a value between 0 and 1 that controls how much weight is given
to the high-flow expert. The soft prediction is:

```text
prediction = (1 - gate) * low_expert_prediction + gate * high_expert_prediction
```

The current configuration uses:

```text
backbone: ConvNeXt-Tiny
core type: shared MoE trunk
adapter: residual MLP
trunk dimension: 256
head dimension: 128
dropout: 0.15
threshold percentile: 55
optimizer: AdamW
learning rate: 0.0001625
weight decay: 0.0012
epochs: 220
patience: 30
batch size: 256
```

The 55th percentile of training discharge is used only to create auxiliary
low/high-flow labels for training the gate. Samples at or below the training-set
55th percentile receive a low-flow gate label, and samples above this percentile
receive a high-flow gate label:

```text
discharge <= training p55 -> low-flow gate label
discharge >  training p55 -> high-flow gate label
```

This should not be interpreted as a physical hydrologic boundary between low and
high flow. Instead, it is a modeling device that gives the gate a weak
supervision signal during training. The gate learns whether an image embedding
is more consistent with the low-flow expert or the high-flow expert.

The threshold is computed from the filtered image-discharge training pairs, not
from the raw USGS discharge time series. The original discharge record is
retained for auditability, but images removed by the quality filter are not used
as modeling samples, so their paired discharge values are also absent from the
training set. Therefore, the p55 threshold describes the clean paired training
dataset used by the model.

At prediction time, the model does not make a hard decision based directly on
the 55th percentile. It predicts a continuous gate value between 0 and 1 and
uses that value to softly combine the low- and high-flow expert predictions.

The model is trained on log-transformed discharge using a Smooth L1 loss. This
reduces the influence of very large discharge values while still preserving
relative differences across the flow range.

The training objective combines:

```text
main regression loss on the mixed prediction
auxiliary gate classification loss
auxiliary low-expert and high-expert losses
auxiliary g-star loss for the ideal soft gate value
```

Current loss weights are:

```text
gate_aux = 0.30
expert_aux = 0.60
gstar_aux = 0.25
high_weight = 1.00
```

The model is trained with multiple random seeds. Predictions from seed-specific
models are combined using a validation-weighted ensemble. This reduces
sensitivity to a single random initialization.

The output includes both:

```text
soft prediction: weighted combination of low/high experts
hard prediction: choose low or high expert using gate >= 0.5
```

The soft prediction is usually the primary model output because it allows a
continuous transition between flow regimes.

## 8. Random Forest and SVM Baselines

The baseline models are trained on the same ConvNeXt embeddings used by the MoE.
They are not trained only on the best MoE ROI. Instead, for every camera and for
each ROI variant, the pipeline trains:

```text
Random Forest
Support Vector Machine
```

Thus RF and SVM are evaluated separately for p05, p50, and p95 embeddings. This
keeps the baseline comparison fair: each model type sees the same visual
features for each ROI condition.

The Random Forest uses 500 trees, square-root feature sampling, and a minimum
leaf size of 2. The SVM baseline uses a standardized feature pipeline followed
by a linear support-vector regressor.

## 9. Final Model Comparison

For each camera, the final comparison is performed in two stages.

First, within each model family, performance is compared across the three ROI
variants:

```text
MoE p05 vs MoE p50 vs MoE p95
RF p05 vs RF p50 vs RF p95
SVM p05 vs SVM p50 vs SVM p95
```

Then, the best ROI result from each model family is compared:

```text
best MoE vs best RF vs best SVM
```

The primary metrics are:

```text
R2
RMSE
MAE
```

For the MoE model, both soft and hard predictions are reported. The soft R2 and
soft RMSE are generally the preferred metrics because the soft mixture better
represents gradual transitions between lower-flow and higher-flow visual states.

## 10. Interpretation Notes

This pipeline is designed to use camera images as indirect observations of
hydraulic state. The model does not directly measure water depth or velocity.
Instead, it learns statistical relationships between visual cues and observed
discharge at the paired USGS gage.

The quality filter and ROI design are therefore central to the validity of the
experiment. If images contain snow-covered channels, ice-covered water, strong
glare, or excessive bank vegetation, the visual signal may no longer correspond
to the open-water discharge process. Similarly, if the ROI includes large
amounts of non-water area, the visual embedding may encode bank or vegetation
conditions rather than hydraulic state.

The three-ROI experiment is included to make the ROI choice explicit and
quantifiable. Rather than assuming one manually drawn ROI is always optimal, the
pipeline evaluates low-flow, median-flow, and high-flow ROI definitions and
reports their effect on prediction performance.
