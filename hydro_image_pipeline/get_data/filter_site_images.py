from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
QUALITY_DERIVED_COLS = {
    "read_ok",
    "width",
    "height",
    "gray_mean",
    "gray_std",
    "entropy",
    "lap_var",
    "grayscale_score",
    "colorfulness",
    "brightness_peak_v",
    "bright_low_sat_ratio",
    "glare_very_bright_low_sat_ratio",
    "glare_saturated_ratio",
    "snow_white_low_sat_ratio",
    "snow_bright_loose_ratio",
    "snow_cool_bright_ratio",
    "snow_blue_dominant_ratio",
    "gray_ice_low_sat_ratio",
    "raindrop_blob_count",
    "raindrop_blob_area_ratio",
    "raindrop_circularity_mean",
    "soft_flag_count",
}


@dataclass
class QualityConfig:
    test_water_year: int = 2025
    fallback_val_ratio: float = 0.2
    min_images_per_water_year: int = 1

    use_local_daytime_filter: bool = True
    use_site_day_windows: bool = True
    local_day_start_hour: float = 8.0
    local_day_end_hour: float = 18.0
    night_grayscale_score: float = 0.02
    night_dark_gray_mean: float = 75.0
    night_dark_peak_v: float = 0.42

    dark_gray_mean_floor: float = 85.0
    severe_dark_gray_mean: float = 55.0

    low_contrast_gray_std_floor: float = 26.0
    severe_low_contrast_gray_std: float = 18.0

    blur_lap_var_floor: float = 250.0
    severe_blur_lap_var: float = 160.0
    blur_quantile: float = 0.10
    blur_threshold_multiplier: float = 0.65

    low_colorfulness_floor: float = 10.0
    severe_low_colorfulness: float = 7.0

    rain_peak_v_threshold: float = 0.60
    raindrop_v_thresh: float = 0.88
    raindrop_s_thresh: float = 0.22
    raindrop_min_blob_count: int = 6
    raindrop_min_blob_area_ratio: float = 0.0008
    raindrop_max_blob_area_ratio: float = 0.06
    raindrop_min_bright_low_sat_ratio: float = 0.0015

    drop_rain_candidates: bool = True
    drop_wet_lens_candidates: bool = True
    drop_fog_haze_candidates: bool = True
    drop_obscured_candidates: bool = True
    drop_glare_candidates: bool = True
    drop_dark_candidates: bool = True
    drop_blurry_candidates: bool = True
    drop_snow_ice_candidates: bool = True

    wet_lens_min_blob_count: int = 6
    wet_lens_min_blob_area_ratio: float = 0.0008
    wet_lens_min_bright_low_sat_ratio: float = 0.0015
    wet_lens_low_contrast_gray_std_ceiling: float = 35.0
    wet_lens_contrast_threshold_multiplier: float = 1.10

    fog_gray_mean_floor: float = 60.0
    fog_gray_std_ceiling: float = 32.0
    fog_contrast_threshold_multiplier: float = 1.05
    fog_colorfulness_ceiling: float = 14.0

    obscured_bright_low_sat_ratio: float = 0.10
    obscured_gray_std_ceiling: float = 42.0
    obscured_colorfulness_ceiling: float = 18.0

    glare_v_thresh: float = 0.96
    glare_s_thresh: float = 0.18
    glare_very_bright_low_sat_ratio: float = 0.16
    glare_saturated_ratio: float = 0.15
    glare_bright_low_sat_ratio: float = 0.16
    glare_peak_v_threshold: float = 0.88

    snow_v_thresh: float = 0.72
    snow_s_thresh: float = 0.28
    snow_white_low_sat_ratio: float = 0.30
    snow_large_white_low_sat_ratio: float = 0.25
    snow_strong_white_low_sat_ratio: float = 0.50
    snow_bright_loose_v_thresh: float = 0.45
    snow_bright_loose_s_thresh: float = 0.55
    snow_bright_loose_ratio: float = 0.65
    snow_bright_loose_gray_mean_floor: float = 90.0
    snow_bright_loose_gray_std_ceiling: float = 35.0
    snow_cool_v_thresh: float = 0.45
    snow_cool_s_thresh: float = 0.55
    snow_cool_bright_ratio: float = 0.55
    snow_blue_v_thresh: float = 0.35
    snow_blue_s_floor: float = 0.30
    snow_blue_margin: float = 0.12
    snow_blue_dominant_ratio: float = 0.85
    snow_blue_gray_mean_floor: float = 80.0
    snow_blue_gray_std_ceiling: float = 40.0
    snow_gray_mean_floor: float = 105.0
    snow_cool_gray_mean_floor: float = 95.0
    snow_cool_gray_std_ceiling: float = 42.0
    snow_colorfulness_ceiling: float = 22.0
    snow_loose_colorfulness_ceiling: float = 28.0
    ice_v_thresh: float = 0.45
    ice_s_thresh: float = 0.18
    ice_gray_low_sat_ratio: float = 0.55
    ice_gray_mean_floor: float = 95.0
    ice_gray_std_ceiling: float = 38.0
    ice_colorfulness_ceiling: float = 35.0
    snow_months: str = "11,12,1,2,3,4"

    max_soft_flags_before_drop: int = 3


SITE_TIMEZONES: dict[str, str] = {
    "01312000": "America/New_York",
    "05124000": "America/Chicago",
    "02094659": "America/New_York",
    "05427880": "America/Chicago",
    "06309000": "America/Denver",
    "14243000": "America/Los_Angeles",
    "06799350": "America/Chicago",
}


SITE_DAY_WINDOWS: dict[str, tuple[float, float]] = {
    "01312000": (8.0, 18.0),
    "05124000": (8.0, 18.5),
    "02094659": (8.0, 18.5),
    "05427880": (8.0, 19.0),
    "06309000": (8.0, 19.0),
    "14243000": (8.0, 19.0),
    "06799350": (8.0, 18.0),
}


def normalize_site_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit() and len(text) < 8:
        text = text.zfill(8)
    return text


def strip_previous_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        col
        for col in df.columns
        if col in QUALITY_DERIVED_COLS
        or col.startswith("quality_")
        or col.startswith("flag_")
        or col.endswith("_threshold")
    ]
    return df.drop(columns=drop_cols, errors="ignore")


class ProgressBar:
    def __init__(self, total: int, label: str, enabled: bool = True) -> None:
        self.total = max(0, int(total))
        self.label = label
        self.enabled = enabled and self.total > 0
        self.current = 0
        self.started_at = time.time()
        self.last_render = 0.0
        if self.enabled:
            self.render(force=True)

    def update(self, n: int = 1) -> None:
        if not self.enabled:
            return
        self.current = min(self.total, self.current + int(n))
        self.render(force=self.current >= self.total)

    def render(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_render < 2.0:
            return

        self.last_render = now
        ratio = self.current / self.total if self.total else 1.0
        filled = int(round(24 * ratio))
        bar = "#" * filled + "-" * (24 - filled)
        elapsed = max(0.0, now - self.started_at)
        rate = self.current / elapsed if elapsed > 0 else 0.0
        sys.stderr.write(
            f"\r{self.label}: [{bar}] {self.current}/{self.total} ({ratio * 100:5.1f}%) {rate:,.1f}/s"
        )
        sys.stderr.flush()

    def close(self) -> None:
        if not self.enabled:
            return
        self.current = self.total
        self.render(force=True)
        sys.stderr.write("\n")
        sys.stderr.flush()


def progress_iter(items, total: int, label: str, enabled: bool = True):
    progress = ProgressBar(total=total, label=label, enabled=enabled)
    try:
        for item in items:
            yield item
            progress.update()
    finally:
        progress.close()


def water_year(ts: pd.Timestamp) -> int:
    return int(ts.year + 1) if ts.month >= 10 else int(ts.year)


def summarize_water_years(matched_valid: pd.DataFrame, min_images_per_water_year: int) -> pd.DataFrame:
    df = matched_valid.copy()
    df["image_time"] = pd.to_datetime(df["image_time"], utc=True, errors="coerce")
    df["water_year"] = df["image_time"].map(water_year)
    df["month_in_wy"] = ((df["image_time"].dt.month + 2) % 12) + 1

    summaries = []
    for wy, chunk in df.groupby("water_year"):
        months = sorted(chunk["month_in_wy"].dropna().astype(int).unique().tolist())
        summaries.append(
            {
                "water_year": int(wy),
                "n_images": int(len(chunk)),
                "months_present": "|".join(str(m) for m in months),
                "n_months_present": int(len(months)),
                "is_complete_water_year": bool(len(months) == 12 and len(chunk) >= min_images_per_water_year),
                "image_time_min": chunk["image_time"].min().isoformat(),
                "image_time_max": chunk["image_time"].max().isoformat(),
            }
        )
    return pd.DataFrame(summaries).sort_values("water_year").reset_index(drop=True)


def assign_splits_by_water_year(
    matched_valid: pd.DataFrame,
    summary: pd.DataFrame,
    test_water_year: int,
    fallback_val_ratio: float,
) -> tuple[pd.DataFrame, dict]:
    complete_years = summary.loc[summary["is_complete_water_year"], "water_year"].astype(int).tolist()
    if test_water_year not in complete_years:
        raise ValueError(f"required test water year WY{test_water_year} is not complete after filtering")

    train_candidate_years = [wy for wy in complete_years if wy < test_water_year]
    if not train_candidate_years:
        raise ValueError(f"no complete train water year found before WY{test_water_year} after filtering")

    df = matched_valid.copy()
    df["image_time"] = pd.to_datetime(df["image_time"], utc=True, errors="coerce")
    df["water_year"] = df["image_time"].map(water_year)
    df = df[df["water_year"].isin(train_candidate_years + [test_water_year])].copy()
    df["split"] = ""

    split_details: dict[str, object] = {"test_water_year": int(test_water_year)}
    df.loc[df["water_year"] == test_water_year, "split"] = "test"

    if len(train_candidate_years) >= 2:
        val_wy = train_candidate_years[-1]
        train_wys = train_candidate_years[:-1]
        df.loc[df["water_year"].isin(train_wys), "split"] = "train"
        df.loc[df["water_year"] == val_wy, "split"] = "val"
        split_details.update(
            {
                "split_strategy": "full_water_years",
                "train_water_years": [int(x) for x in train_wys],
                "val_water_year": int(val_wy),
            }
        )
    else:
        only_train_wy = train_candidate_years[0]
        wy_chunk = df[df["water_year"] == only_train_wy].sort_values("image_time").copy()
        n_val = max(1, int(np.ceil(len(wy_chunk) * float(fallback_val_ratio))))
        val_idx = wy_chunk.tail(n_val).index
        train_idx = wy_chunk.head(max(0, len(wy_chunk) - n_val)).index
        df.loc[train_idx, "split"] = "train"
        df.loc[val_idx, "split"] = "val"
        split_details.update(
            {
                "split_strategy": "single_train_water_year_with_time_tail_val",
                "train_water_years": [int(only_train_wy)],
                "val_water_year": int(only_train_wy),
                "fallback_val_ratio": float(fallback_val_ratio),
            }
        )

    split_counts = df["split"].value_counts().to_dict()
    if "train" not in split_counts or "val" not in split_counts or "test" not in split_counts:
        raise ValueError(f"split assignment failed after filtering, got counts: {split_counts}")

    split_details["split_counts"] = {k: int(v) for k, v in split_counts.items()}
    return df.sort_values("image_time").reset_index(drop=True), split_details


def compute_grayscale_score(image_bgr: np.ndarray) -> float:
    b = image_bgr[..., 0].astype(np.float32)
    g = image_bgr[..., 1].astype(np.float32)
    r = image_bgr[..., 2].astype(np.float32)
    diff = (np.abs(r - g) + np.abs(r - b) + np.abs(g - b)) / 3.0
    return float(np.mean(diff)) / 255.0


def compute_colorfulness(image_bgr: np.ndarray) -> float:
    b, g, r = cv2.split(image_bgr.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)
    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))


def compute_entropy(gray: np.ndarray) -> float:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float32)
    p = hist / (hist.sum() + 1e-8)
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def brightness_peak_v(image_bgr: np.ndarray, bins: int = 50) -> float:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[..., 2] / 255.0
    hist, bin_edges = np.histogram(v.reshape(-1), bins=bins, range=(0.0, 1.0))
    peak_bin = int(np.argmax(hist))
    return float((bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0)


def raindrop_features(
    image_bgr: np.ndarray,
    v_thresh: float,
    s_thresh: float,
    open_kernel: int = 3,
    min_area: int = 30,
    max_area: int = 4000,
    min_circularity: float = 0.35,
) -> dict[str, float | int]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0

    mask = ((v > v_thresh) & (s < s_thresh)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    image_area = float(h * w)

    blob_count = 0
    blob_area_sum = 0.0
    circularity_list: list[float] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        if circularity < min_circularity:
            continue

        blob_count += 1
        blob_area_sum += float(area)
        circularity_list.append(circularity)

    bright_low_sat_ratio = float(np.count_nonzero(mask)) / image_area
    blob_area_ratio = float(blob_area_sum) / image_area
    circularity_mean = float(np.mean(circularity_list)) if circularity_list else 0.0

    return {
        "bright_low_sat_ratio": bright_low_sat_ratio,
        "raindrop_blob_count": int(blob_count),
        "raindrop_blob_area_ratio": blob_area_ratio,
        "raindrop_circularity_mean": circularity_mean,
    }


def is_raindrop_candidate(feat: dict[str, float | int], cfg: QualityConfig) -> bool:
    return bool(
        feat.get("raindrop_blob_count", 0) >= cfg.raindrop_min_blob_count
        and feat.get("raindrop_blob_area_ratio", 0.0) >= cfg.raindrop_min_blob_area_ratio
        and feat.get("raindrop_blob_area_ratio", 0.0) <= cfg.raindrop_max_blob_area_ratio
        and feat.get("bright_low_sat_ratio", 0.0) >= cfg.raindrop_min_bright_low_sat_ratio
    )


def auto_threshold(series: pd.Series, base_floor: float, quantile: float, multiplier: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return base_floor
    return max(base_floor, float(clean.quantile(quantile)) * multiplier)


def compute_local_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    local_hour = pd.Series(np.nan, index=df.index, dtype="float64")
    local_month = pd.Series(np.nan, index=df.index, dtype="float64")
    if "quality_image_time" not in df.columns or "quality_site_id" not in df.columns:
        return pd.DataFrame({"quality_local_hour": local_hour, "quality_local_month": local_month})

    ts_utc = pd.to_datetime(df["quality_image_time"], utc=True, errors="coerce")
    site_ids = df["quality_site_id"].map(normalize_site_id)
    for site_id in sorted(site_ids.dropna().unique().tolist()):
        tz_name = SITE_TIMEZONES.get(site_id)
        if not tz_name:
            continue

        mask = site_ids == site_id
        local_ts = ts_utc.loc[mask].dt.tz_convert(tz_name)
        local_hour.loc[mask] = (
            local_ts.dt.hour.astype(float)
            + local_ts.dt.minute.astype(float) / 60.0
            + local_ts.dt.second.astype(float) / 3600.0
        )
        local_month.loc[mask] = local_ts.dt.month.astype(float)
    return pd.DataFrame({"quality_local_hour": local_hour, "quality_local_month": local_month})


def compute_site_day_windows(df: pd.DataFrame, cfg: QualityConfig) -> tuple[pd.Series, pd.Series]:
    start_hour = pd.Series(float(cfg.local_day_start_hour), index=df.index, dtype="float64")
    end_hour = pd.Series(float(cfg.local_day_end_hour), index=df.index, dtype="float64")
    if not cfg.use_site_day_windows or "quality_site_id" not in df.columns:
        return start_hour, end_hour

    site_ids = df["quality_site_id"].map(normalize_site_id)
    for site_id, (site_start, site_end) in SITE_DAY_WINDOWS.items():
        mask = site_ids == site_id
        if mask.any():
            start_hour.loc[mask] = float(site_start)
            end_hour.loc[mask] = float(site_end)
    return start_hour, end_hour


def parse_months(months: str) -> set[int]:
    parsed: set[int] = set()
    for item in str(months).split(","):
        item = item.strip()
        if not item:
            continue
        month = int(item)
        if month < 1 or month > 12:
            raise ValueError(f"invalid month in snow_months: {item}")
        parsed.add(month)
    return parsed


def load_quality_metrics(image_path: Path, cfg: QualityConfig) -> dict[str, object]:
    row: dict[str, object] = {"image_path": str(image_path.resolve()), "read_ok": False}
    img = cv2.imread(str(image_path))
    if img is None:
        return row

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = gray.shape[:2]
    v = hsv[..., 2].astype(np.float32) / 255.0
    s = hsv[..., 1].astype(np.float32) / 255.0
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    rain_feat = raindrop_features(
        img,
        v_thresh=cfg.raindrop_v_thresh,
        s_thresh=cfg.raindrop_s_thresh,
    )

    row.update(
        {
            "read_ok": True,
            "width": int(w),
            "height": int(h),
            "gray_mean": float(gray.mean()),
            "gray_std": float(gray.std()),
            "entropy": compute_entropy(gray),
            "lap_var": lap_var,
            "grayscale_score": compute_grayscale_score(img),
            "colorfulness": compute_colorfulness(img),
            "brightness_peak_v": brightness_peak_v(img),
            "bright_low_sat_ratio": float(np.mean((v > cfg.raindrop_v_thresh) & (s < cfg.raindrop_s_thresh))),
            "glare_very_bright_low_sat_ratio": float(np.mean((v > cfg.glare_v_thresh) & (s < cfg.glare_s_thresh))),
            "glare_saturated_ratio": float(np.mean((v > 0.985) & (s < 0.30))),
            "snow_white_low_sat_ratio": float(np.mean((v > cfg.snow_v_thresh) & (s < cfg.snow_s_thresh))),
            "snow_bright_loose_ratio": float(
                np.mean((v > cfg.snow_bright_loose_v_thresh) & (s < cfg.snow_bright_loose_s_thresh))
            ),
            "snow_cool_bright_ratio": float(
                np.mean(
                    (v > cfg.snow_cool_v_thresh)
                    & (s < cfg.snow_cool_s_thresh)
                    & (b >= r - 0.03)
                    & (g >= r - 0.10)
                )
            ),
            "snow_blue_dominant_ratio": float(
                np.mean(
                    (v > cfg.snow_blue_v_thresh)
                    & (s > cfg.snow_blue_s_floor)
                    & (b >= r + cfg.snow_blue_margin)
                    & (g >= r - 0.05)
                )
            ),
            "gray_ice_low_sat_ratio": float(np.mean((v > cfg.ice_v_thresh) & (s < cfg.ice_s_thresh))),
            **rain_feat,
        }
    )
    return row


def classify_quality(metrics: pd.DataFrame, cfg: QualityConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    df = metrics.copy()

    local_time_parts = compute_local_time_parts(df)
    df["quality_local_hour"] = local_time_parts["quality_local_hour"]
    df["quality_local_month"] = local_time_parts["quality_local_month"]
    df["quality_day_start_hour"], df["quality_day_end_hour"] = compute_site_day_windows(df, cfg)
    has_local_hour = df["quality_local_hour"].notna()
    df["flag_time_night"] = False
    if cfg.use_local_daytime_filter:
        df["flag_time_night"] = has_local_hour & (
            (df["quality_local_hour"] < df["quality_day_start_hour"])
            | (df["quality_local_hour"] >= df["quality_day_end_hour"])
        )

    readable = df[df["read_ok"].fillna(False)].copy()
    readable_non_night = readable[
        (readable["grayscale_score"] >= cfg.night_grayscale_score)
        & ~readable["flag_time_night"].fillna(False)
        & ~(
            (readable["gray_mean"].fillna(0.0) < cfg.night_dark_gray_mean)
            & (readable["brightness_peak_v"].fillna(0.0) < cfg.night_dark_peak_v)
        )
    ].copy()

    dark_threshold = auto_threshold(readable_non_night["gray_mean"], cfg.dark_gray_mean_floor, 0.05, 0.90)
    contrast_threshold = auto_threshold(
        readable_non_night["gray_std"], cfg.low_contrast_gray_std_floor, 0.05, 0.75
    )
    blur_threshold = auto_threshold(
        readable_non_night["lap_var"],
        cfg.blur_lap_var_floor,
        cfg.blur_quantile,
        cfg.blur_threshold_multiplier,
    )
    fog_contrast_threshold = max(
        cfg.fog_gray_std_ceiling,
        float(contrast_threshold) * cfg.fog_contrast_threshold_multiplier,
    )
    wet_lens_contrast_threshold = max(
        cfg.wet_lens_low_contrast_gray_std_ceiling,
        float(contrast_threshold) * cfg.wet_lens_contrast_threshold_multiplier,
    )

    color_clean = pd.to_numeric(readable_non_night["colorfulness"], errors="coerce").dropna()
    if color_clean.empty:
        low_color_threshold = cfg.low_colorfulness_floor
    else:
        low_color_threshold = max(
            cfg.severe_low_colorfulness,
            min(cfg.low_colorfulness_floor, float(color_clean.quantile(0.10))),
        )

    df["flag_unreadable"] = ~df["read_ok"].fillna(False)
    df["flag_pixel_night"] = df["grayscale_score"].fillna(0.0) < cfg.night_grayscale_score
    df["flag_dark_night"] = (df["gray_mean"].fillna(0.0) < cfg.night_dark_gray_mean) & (
        df["brightness_peak_v"].fillna(0.0) < cfg.night_dark_peak_v
    )
    df["flag_night"] = df["flag_pixel_night"] | df["flag_time_night"] | df["flag_dark_night"]
    df["flag_dark"] = df["gray_mean"].fillna(0.0) < dark_threshold
    df["flag_severe_dark"] = df["gray_mean"].fillna(0.0) < cfg.severe_dark_gray_mean
    df["flag_low_contrast"] = df["gray_std"].fillna(0.0) < contrast_threshold
    df["flag_severe_low_contrast"] = df["gray_std"].fillna(0.0) < cfg.severe_low_contrast_gray_std
    df["flag_blurry"] = df["lap_var"].fillna(0.0) < blur_threshold
    df["flag_severe_blurry"] = df["lap_var"].fillna(0.0) < cfg.severe_blur_lap_var
    df["flag_low_colorfulness"] = df["colorfulness"].fillna(0.0) < low_color_threshold
    df["flag_severe_low_colorfulness"] = df["colorfulness"].fillna(0.0) < cfg.severe_low_colorfulness
    df["flag_rain_peak"] = df["brightness_peak_v"].fillna(0.0) > cfg.rain_peak_v_threshold
    df["flag_rain_blob"] = df.apply(lambda row: is_raindrop_candidate(row.to_dict(), cfg), axis=1)
    df["flag_rain_candidate"] = df["flag_rain_peak"] & df["flag_rain_blob"]
    df["flag_wet_lens_candidate"] = (
        (df["raindrop_blob_count"].fillna(0) >= cfg.wet_lens_min_blob_count)
        & (df["raindrop_blob_area_ratio"].fillna(0.0) >= cfg.wet_lens_min_blob_area_ratio)
        & (df["bright_low_sat_ratio"].fillna(0.0) >= cfg.wet_lens_min_bright_low_sat_ratio)
        & (
            df["flag_rain_peak"]
            | (df["gray_std"].fillna(np.inf) < wet_lens_contrast_threshold)
        )
    )
    df["flag_fog_haze_candidate"] = (
        (df["gray_mean"].fillna(0.0) >= cfg.fog_gray_mean_floor)
        & (df["gray_std"].fillna(np.inf) < fog_contrast_threshold)
        & (df["colorfulness"].fillna(np.inf) < cfg.fog_colorfulness_ceiling)
    )
    df["flag_obscured_candidate"] = (
        (df["bright_low_sat_ratio"].fillna(0.0) >= cfg.obscured_bright_low_sat_ratio)
        & (df["gray_std"].fillna(np.inf) < cfg.obscured_gray_std_ceiling)
        & (df["colorfulness"].fillna(np.inf) < cfg.obscured_colorfulness_ceiling)
    )
    df["flag_glare_candidate"] = (
        (
            (df["glare_very_bright_low_sat_ratio"].fillna(0.0) >= cfg.glare_very_bright_low_sat_ratio)
            & (df["brightness_peak_v"].fillna(0.0) >= cfg.glare_peak_v_threshold)
        )
        | (
            (df["glare_saturated_ratio"].fillna(0.0) >= cfg.glare_saturated_ratio)
            & (df["bright_low_sat_ratio"].fillna(0.0) >= cfg.glare_bright_low_sat_ratio)
        )
    )
    snow_months = parse_months(cfg.snow_months)
    df["flag_snow_month"] = df["quality_local_month"].isin(snow_months)
    df["flag_snow_ice_candidate"] = df["flag_snow_month"] & (
        (
            (df["snow_white_low_sat_ratio"].fillna(0.0) >= cfg.snow_white_low_sat_ratio)
            & (df["gray_mean"].fillna(0.0) >= cfg.snow_gray_mean_floor)
            & (df["colorfulness"].fillna(np.inf) <= cfg.snow_colorfulness_ceiling)
        )
        | (
            (df["snow_white_low_sat_ratio"].fillna(0.0) >= cfg.snow_large_white_low_sat_ratio)
            & (df["gray_mean"].fillna(0.0) >= cfg.snow_gray_mean_floor)
            & (df["colorfulness"].fillna(np.inf) <= cfg.snow_loose_colorfulness_ceiling)
        )
        | (
            (df["snow_white_low_sat_ratio"].fillna(0.0) >= cfg.snow_strong_white_low_sat_ratio)
            & (df["gray_mean"].fillna(0.0) >= cfg.snow_gray_mean_floor)
            & (df["colorfulness"].fillna(np.inf) <= cfg.snow_loose_colorfulness_ceiling)
        )
        | (
            (df["snow_bright_loose_ratio"].fillna(0.0) >= cfg.snow_bright_loose_ratio)
            & (df["gray_mean"].fillna(0.0) >= cfg.snow_bright_loose_gray_mean_floor)
            & (df["gray_std"].fillna(np.inf) <= cfg.snow_bright_loose_gray_std_ceiling)
        )
        | (
            (df["snow_cool_bright_ratio"].fillna(0.0) >= cfg.snow_cool_bright_ratio)
            & (df["gray_mean"].fillna(0.0) >= cfg.snow_cool_gray_mean_floor)
            & (df["gray_std"].fillna(np.inf) <= cfg.snow_cool_gray_std_ceiling)
        )
        | (
            (df["snow_blue_dominant_ratio"].fillna(0.0) >= cfg.snow_blue_dominant_ratio)
            & (df["gray_mean"].fillna(0.0) >= cfg.snow_blue_gray_mean_floor)
            & (df["gray_std"].fillna(np.inf) <= cfg.snow_blue_gray_std_ceiling)
        )
        | (
            (df["gray_ice_low_sat_ratio"].fillna(0.0) >= cfg.ice_gray_low_sat_ratio)
            & (df["gray_mean"].fillna(0.0) >= cfg.ice_gray_mean_floor)
            & (df["gray_std"].fillna(np.inf) <= cfg.ice_gray_std_ceiling)
            & (df["colorfulness"].fillna(np.inf) <= cfg.ice_colorfulness_ceiling)
        )
    )

    moderate_flag_cols = [
        "flag_dark",
        "flag_low_contrast",
        "flag_blurry",
        "flag_low_colorfulness",
        "flag_rain_candidate",
        "flag_wet_lens_candidate",
        "flag_fog_haze_candidate",
        "flag_obscured_candidate",
        "flag_glare_candidate",
        "flag_snow_ice_candidate",
    ]
    df["soft_flag_count"] = df[moderate_flag_cols].astype(int).sum(axis=1)

    drop_reasons: list[list[str]] = []
    keep_mask: list[bool] = []
    review_mask: list[bool] = []
    for row in df.to_dict(orient="records"):
        reasons: list[str] = []
        keep = True
        review = False

        if row["flag_unreadable"]:
            reasons.append("unreadable")
            keep = False
        if row["flag_time_night"]:
            reasons.append("local_night_time")
            keep = False
        if row["flag_pixel_night"]:
            reasons.append("night_or_near_grayscale")
            keep = False
        if row["flag_dark_night"]:
            reasons.append("night_dark_scene")
            keep = False

        if row["flag_severe_dark"]:
            reasons.append("severe_dark")
            keep = False
        if row["flag_severe_blurry"]:
            reasons.append("severe_blurry")
            keep = False

        if cfg.drop_dark_candidates and row["flag_dark"]:
            reasons.append("dark_candidate")
            keep = False
        if cfg.drop_blurry_candidates and row["flag_blurry"]:
            reasons.append("blurry_candidate")
            keep = False
        if cfg.drop_rain_candidates and row["flag_rain_candidate"]:
            reasons.append("rain_or_wet_lens_candidate")
            keep = False
        if cfg.drop_wet_lens_candidates and row["flag_wet_lens_candidate"]:
            reasons.append("wet_lens_or_water_drops")
            keep = False
        if cfg.drop_fog_haze_candidates and row["flag_fog_haze_candidate"]:
            reasons.append("fog_haze_or_water_vapor")
            keep = False
        if cfg.drop_obscured_candidates and row["flag_obscured_candidate"]:
            reasons.append("obscured_or_washed_out")
            keep = False
        if cfg.drop_glare_candidates and row["flag_glare_candidate"]:
            reasons.append("strong_glare_or_overexposure")
            keep = False
        if cfg.drop_snow_ice_candidates and row["flag_snow_ice_candidate"]:
            reasons.append("snow_or_ice_candidate")
            keep = False

        if row["flag_dark"] and row["flag_low_contrast"]:
            reasons.append("dark_and_low_contrast")
            keep = False
        if row["flag_blurry"] and row["flag_low_contrast"]:
            reasons.append("blurry_and_low_contrast")
            keep = False
        if row["flag_low_colorfulness"] and row["flag_dark"]:
            reasons.append("low_colorfulness_and_dark")
            keep = False

        if row["soft_flag_count"] >= cfg.max_soft_flags_before_drop:
            reasons.append("too_many_soft_quality_flags")
            keep = False

        if keep and row["flag_rain_candidate"]:
            reasons.append("review_rain_candidate")
            review = True
        if keep and row["flag_wet_lens_candidate"]:
            reasons.append("review_wet_lens_or_water_drops")
            review = True
        if keep and row["flag_fog_haze_candidate"]:
            reasons.append("review_fog_haze_or_water_vapor")
            review = True
        if keep and row["flag_obscured_candidate"]:
            reasons.append("review_obscured_or_washed_out")
            review = True
        if keep and row["flag_glare_candidate"]:
            reasons.append("review_strong_glare_or_overexposure")
            review = True
        if keep and row["flag_snow_ice_candidate"]:
            reasons.append("review_snow_or_ice_candidate")
            review = True
        if keep and row["flag_low_colorfulness"] and not row["flag_dark"]:
            reasons.append("review_low_colorfulness")
            review = True

        drop_reasons.append(reasons)
        keep_mask.append(keep)
        review_mask.append(review)

    df["quality_keep"] = keep_mask
    df["quality_review"] = review_mask
    df["quality_reasons"] = ["|".join(items) for items in drop_reasons]

    thresholds = {
        "dark_threshold": dark_threshold,
        "contrast_threshold": contrast_threshold,
        "blur_threshold": blur_threshold,
        "low_colorfulness_threshold": low_color_threshold,
        "fog_contrast_threshold": fog_contrast_threshold,
        "wet_lens_contrast_threshold": wet_lens_contrast_threshold,
        "local_day_start_hour": float(cfg.local_day_start_hour),
        "local_day_end_hour": float(cfg.local_day_end_hour),
        "use_site_day_windows": bool(cfg.use_site_day_windows),
        "site_day_windows": {
            site_id: {"start": float(start), "end": float(end)}
            for site_id, (start, end) in SITE_DAY_WINDOWS.items()
        },
        "night_dark_gray_mean": float(cfg.night_dark_gray_mean),
        "night_dark_peak_v": float(cfg.night_dark_peak_v),
        "blur_quantile": float(cfg.blur_quantile),
        "blur_threshold_multiplier": float(cfg.blur_threshold_multiplier),
        "snow_white_low_sat_ratio": float(cfg.snow_white_low_sat_ratio),
        "snow_large_white_low_sat_ratio": float(cfg.snow_large_white_low_sat_ratio),
        "snow_strong_white_low_sat_ratio": float(cfg.snow_strong_white_low_sat_ratio),
        "snow_bright_loose_ratio": float(cfg.snow_bright_loose_ratio),
        "snow_bright_loose_v_thresh": float(cfg.snow_bright_loose_v_thresh),
        "snow_bright_loose_s_thresh": float(cfg.snow_bright_loose_s_thresh),
        "snow_cool_bright_ratio": float(cfg.snow_cool_bright_ratio),
        "snow_cool_v_thresh": float(cfg.snow_cool_v_thresh),
        "snow_cool_s_thresh": float(cfg.snow_cool_s_thresh),
        "snow_blue_dominant_ratio": float(cfg.snow_blue_dominant_ratio),
        "snow_blue_v_thresh": float(cfg.snow_blue_v_thresh),
        "snow_blue_s_floor": float(cfg.snow_blue_s_floor),
        "snow_blue_margin": float(cfg.snow_blue_margin),
        "snow_gray_mean_floor": float(cfg.snow_gray_mean_floor),
        "snow_bright_loose_gray_mean_floor": float(cfg.snow_bright_loose_gray_mean_floor),
        "snow_bright_loose_gray_std_ceiling": float(cfg.snow_bright_loose_gray_std_ceiling),
        "snow_cool_gray_mean_floor": float(cfg.snow_cool_gray_mean_floor),
        "snow_cool_gray_std_ceiling": float(cfg.snow_cool_gray_std_ceiling),
        "snow_blue_gray_mean_floor": float(cfg.snow_blue_gray_mean_floor),
        "snow_blue_gray_std_ceiling": float(cfg.snow_blue_gray_std_ceiling),
        "snow_colorfulness_ceiling": float(cfg.snow_colorfulness_ceiling),
        "ice_gray_low_sat_ratio": float(cfg.ice_gray_low_sat_ratio),
        "ice_gray_mean_floor": float(cfg.ice_gray_mean_floor),
        "ice_gray_std_ceiling": float(cfg.ice_gray_std_ceiling),
        "ice_colorfulness_ceiling": float(cfg.ice_colorfulness_ceiling),
        "snow_months": cfg.snow_months,
        "glare_very_bright_low_sat_ratio": float(cfg.glare_very_bright_low_sat_ratio),
        "glare_saturated_ratio": float(cfg.glare_saturated_ratio),
        "glare_bright_low_sat_ratio": float(cfg.glare_bright_low_sat_ratio),
        "glare_peak_v_threshold": float(cfg.glare_peak_v_threshold),
    }
    return df, thresholds


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path

    for i in range(1, 100000):
        candidate = path.with_name(f"{path.stem}__filtered_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"could not find an unused destination name near {path}")


def iter_image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def sync_images_all_to_kept(
    kept: pd.DataFrame,
    audit: pd.DataFrame,
    site_dir: Path,
    filtered_dir_name: str,
    show_progress: bool,
) -> dict[str, object]:
    images_all_dir = site_dir / "images_all"
    filtered_dir = site_dir / filtered_dir_name
    filtered_dir.mkdir(parents=True, exist_ok=True)

    kept_paths = {Path(str(path)).resolve() for path in kept["image_path"].astype(str).tolist()}
    quality_dropped_paths = {
        Path(str(path)).resolve()
        for path in audit.loc[~audit["quality_keep"].fillna(False), "image_path"].astype(str).tolist()
    }

    image_files = iter_image_files(images_all_dir)
    to_move = [path for path in image_files if path.resolve() not in kept_paths]

    moved_rows: list[dict[str, object]] = []
    n_quality_dropped_moved = 0
    n_extra_moved = 0
    for src in progress_iter(
        to_move,
        total=len(to_move),
        label=f"{site_dir.name} move filtered images",
        enabled=show_progress,
    ):
        src_resolved = src.resolve()
        dst = unique_destination(filtered_dir / src.name)
        shutil.move(str(src), str(dst))
        is_quality_dropped = src_resolved in quality_dropped_paths
        n_quality_dropped_moved += int(is_quality_dropped)
        n_extra_moved += int(not is_quality_dropped)
        moved_rows.append(
            {
                "filename": src.name,
                "original_path": str(src_resolved),
                "filtered_path": str(dst.resolve()),
                "source": "quality_filter" if is_quality_dropped else "not_in_kept_matched",
            }
        )

    manifest_path = site_dir / "filtered_image_manifest.csv"
    pd.DataFrame(moved_rows).to_csv(manifest_path, index=False)

    remaining_images_all = len(iter_image_files(images_all_dir))
    return {
        "images_all_dir": str(images_all_dir.resolve()),
        "filtered_dir": str(filtered_dir.resolve()),
        "filtered_image_manifest_csv": str(manifest_path.resolve()),
        "n_images_all_before": int(len(image_files)),
        "n_kept_expected": int(len(kept_paths)),
        "n_moved_to_filtered_dir": int(len(moved_rows)),
        "n_quality_dropped_moved": int(n_quality_dropped_moved),
        "n_not_in_kept_matched_moved": int(n_extra_moved),
        "n_images_all_after": int(remaining_images_all),
    }


def reset_split_dirs(site_dir: Path) -> None:
    for split_name in ("train", "val", "test"):
        split_dir = site_dir / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)


def materialize_split_dirs(split_df: pd.DataFrame, site_dir: Path, show_progress: bool) -> None:
    reset_split_dirs(site_dir)
    rows = split_df.to_dict(orient="records")
    for row in progress_iter(
        rows,
        total=len(rows),
        label=f"{site_dir.name} rebuild train/val/test",
        enabled=show_progress,
    ):
        src = Path(str(row["image_path"])).resolve()
        dst = site_dir / str(row["split"]) / src.name
        hardlink_or_copy(src, dst)


def backup_if_needed(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".pre_filter_backup")
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)


def activate_filtered_outputs(site_dir: Path) -> None:
    replacements = {
        site_dir / "matched_filtered.csv": site_dir / "matched.csv",
        site_dir / "water_year_summary_filtered.csv": site_dir / "water_year_summary.csv",
        site_dir / "split_manifest_filtered.csv": site_dir / "split_manifest.csv",
    }
    for src, dst in replacements.items():
        if not src.exists():
            raise FileNotFoundError(f"missing filtered artifact: {src}")
        backup_if_needed(dst)
        shutil.copy2(src, dst)


def choose_sites(output_root: Path, site_info_csv: Path | None, site_ids: list[str]) -> list[tuple[str, Path]]:
    if site_info_csv is not None:
        site_info = pd.read_csv(site_info_csv, dtype={"site_id": "string"})
        if "site_id" not in site_info.columns:
            raise ValueError("site_info CSV must contain site_id")
        site_info["site_id"] = site_info["site_id"].astype(str).str.strip()
        records = [(sid, output_root / sid) for sid in site_info["site_id"].tolist()]
    else:
        records = [(p.name, p) for p in sorted(output_root.iterdir()) if p.is_dir() and p.name.isdigit()]

    if site_ids:
        wanted = {str(x).strip() for x in site_ids}
        records = [(sid, site_dir) for sid, site_dir in records if sid in wanted]
    return records


def filter_single_site(
    site_id: str,
    site_dir: Path,
    cfg: QualityConfig,
    activate: bool,
    filtered_dir_name: str,
    show_progress: bool,
) -> dict[str, object]:
    matched_path = site_dir / "matched.csv"
    if not matched_path.exists():
        raise FileNotFoundError(f"missing matched.csv for site {site_id}: {matched_path}")

    matched = strip_previous_quality_columns(pd.read_csv(matched_path))
    if "image_path" not in matched.columns:
        raise ValueError(f"{matched_path} must contain image_path")

    image_paths = matched["image_path"].astype(str).tolist()
    metric_rows = [
        load_quality_metrics(Path(p), cfg)
        for p in progress_iter(
            image_paths,
            total=len(image_paths),
            label=f"{site_id} quality scan",
            enabled=show_progress,
        )
    ]
    metrics = pd.DataFrame(metric_rows)
    metadata_cols = ["image_path"]
    if "image_time" in matched.columns:
        metadata_cols.append("image_time")
    if "site_id" in matched.columns:
        metadata_cols.append("site_id")
    metadata = matched[metadata_cols].copy().rename(
        columns={"image_time": "quality_image_time", "site_id": "quality_site_id"}
    )
    if "quality_site_id" not in metadata.columns:
        metadata["quality_site_id"] = str(site_id)
    metrics = metrics.merge(metadata, on="image_path", how="left", validate="one_to_one")
    metrics["quality_site_id"] = normalize_site_id(site_id)
    classified, thresholds = classify_quality(metrics, cfg)

    audit = matched.merge(classified, on="image_path", how="left", validate="one_to_one")
    audit_path = site_dir / "image_quality_audit.csv"
    audit.to_csv(audit_path, index=False)

    kept = audit[audit["quality_keep"].fillna(False)].copy()
    if kept.empty:
        raise ValueError(f"all images were filtered out for site {site_id}")

    matched_output_cols = matched.columns.tolist()
    kept_records = kept[matched_output_cols].copy()
    for col in ("image_time", "hydro_time", "nearest_obs_time"):
        if col in kept_records.columns:
            kept_records[col] = pd.to_datetime(kept_records[col], utc=True, errors="coerce")

    filtered_matched_path = site_dir / "matched_filtered.csv"
    kept_out = kept_records.copy()
    for col in ("image_time", "hydro_time", "nearest_obs_time"):
        if col in kept_out.columns:
            kept_out[col] = kept_out[col].map(lambda x: x.isoformat() if pd.notna(x) else "")
    kept_out.to_csv(filtered_matched_path, index=False)

    summary_df = summarize_water_years(kept_records, cfg.min_images_per_water_year)
    summary_path = site_dir / "water_year_summary_filtered.csv"
    summary_df.to_csv(summary_path, index=False)

    split_df, split_meta = assign_splits_by_water_year(
        matched_valid=kept_records,
        summary=summary_df,
        test_water_year=cfg.test_water_year,
        fallback_val_ratio=cfg.fallback_val_ratio,
    )
    split_path = site_dir / "split_manifest_filtered.csv"
    split_out = split_df.copy()
    for col in ("image_time", "hydro_time", "nearest_obs_time"):
        if col in split_out.columns:
            split_out[col] = pd.to_datetime(split_out[col], utc=True, errors="coerce").map(
                lambda x: x.isoformat() if pd.notna(x) else ""
            )
    split_out.to_csv(split_path, index=False)

    image_file_actions: dict[str, object] = {}
    if activate:
        activate_filtered_outputs(site_dir)
        image_file_actions = sync_images_all_to_kept(
            kept=kept_records,
            audit=audit,
            site_dir=site_dir,
            filtered_dir_name=filtered_dir_name,
            show_progress=show_progress,
        )
        materialize_split_dirs(split_df, site_dir, show_progress=show_progress)

    summary = {
        "site_id": site_id,
        "site_dir": str(site_dir.resolve()),
        "n_input_rows": int(len(matched)),
        "n_kept_rows": int(len(kept)),
        "n_dropped_rows": int(len(matched) - len(kept)),
        "keep_ratio": float(len(kept) / max(1, len(matched))),
        "quality_thresholds": thresholds,
        "drop_reason_counts": {
            str(k): int(v)
            for k, v in audit.loc[~audit["quality_keep"].fillna(False), "quality_reasons"].value_counts().to_dict().items()
        },
        "review_count": int(audit["quality_review"].fillna(False).sum()),
        "split_meta": split_meta,
        "image_file_actions": image_file_actions,
        "files": {
            "audit_csv": str(audit_path.resolve()),
            "matched_filtered_csv": str(filtered_matched_path.resolve()),
            "water_year_summary_filtered_csv": str(summary_path.resolve()),
            "split_manifest_filtered_csv": str(split_path.resolve()),
        },
        "activated": bool(activate),
    }
    (site_dir / "filter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Filter low-quality images after prepare_data.py. "
            "By default this writes *_filtered artifacts only; add --activate-filtered to replace active matched/split files "
            "and move dropped images out of images_all."
        )
    )
    ap.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parents[2] / "data"),
        help="Prepare-data output root containing {site_id}/ folders",
    )
    ap.add_argument("--site-info-csv", default="", help="Optional CSV with site_id column to limit sites")
    ap.add_argument("--site-id", action="append", default=[], help="Optional site_id to filter; can repeat")
    ap.add_argument("--test-water-year", type=int, default=2025, help="Fixed test water year")
    ap.add_argument("--fallback-val-ratio", type=float, default=0.2, help="Fallback val ratio inside single train WY")
    ap.add_argument("--min-images-per-water-year", type=int, default=1, help="Minimum images for a complete WY")
    ap.add_argument(
        "--activate-filtered",
        action="store_true",
        help="Replace matched/split files, move filtered images to a side folder, and rebuild train/val/test",
    )
    ap.add_argument(
        "--filtered-dir-name",
        default="images_filtered_out",
        help="Site-folder directory name where dropped images are moved when --activate-filtered is used",
    )
    ap.add_argument("--no-progress", action="store_true", help="Disable visible progress bars")
    ap.add_argument(
        "--disable-local-daytime-filter",
        action="store_true",
        help="Do not drop images outside the site-specific local daytime window",
    )
    ap.add_argument(
        "--disable-site-day-windows",
        action="store_true",
        help="Use the uniform --local-day-start-hour/--local-day-end-hour window for every site",
    )
    ap.add_argument("--local-day-start-hour", type=float, default=8.0, help="Fallback local hour where usable daytime starts")
    ap.add_argument("--local-day-end-hour", type=float, default=18.0, help="Fallback local hour where usable daytime ends")
    ap.add_argument("--night-grayscale-score", type=float, default=0.02, help="Pixel day/night cutoff")
    ap.add_argument("--night-dark-gray-mean", type=float, default=75.0, help="Very dark scene hard-drop mean threshold")
    ap.add_argument("--night-dark-peak-v", type=float, default=0.42, help="Very dark scene hard-drop brightness peak")
    ap.add_argument("--dark-gray-mean-floor", type=float, default=85.0, help="Absolute floor for dark-image flagging")
    ap.add_argument("--severe-dark-gray-mean", type=float, default=55.0, help="Severe darkness hard-drop threshold")
    ap.add_argument("--low-contrast-gray-std-floor", type=float, default=26.0, help="Absolute floor for low-contrast flagging")
    ap.add_argument("--severe-low-contrast-gray-std", type=float, default=18.0, help="Severe low-contrast threshold")
    ap.add_argument("--blur-lap-var-floor", type=float, default=250.0, help="Absolute floor for blur flagging")
    ap.add_argument("--severe-blur-lap-var", type=float, default=160.0, help="Severe blur hard-drop threshold")
    ap.add_argument("--blur-quantile", type=float, default=0.10, help="Site-adaptive blur quantile")
    ap.add_argument("--blur-threshold-multiplier", type=float, default=0.65, help="Multiplier applied to the blur quantile")
    ap.add_argument("--low-colorfulness-floor", type=float, default=10.0, help="Base low-colorfulness threshold")
    ap.add_argument("--severe-low-colorfulness", type=float, default=7.0, help="Severe low-colorfulness threshold")
    ap.add_argument("--keep-rain-candidates", dest="drop_rain_candidates", action="store_false", help="Review rain candidates instead of dropping them")
    ap.add_argument("--keep-wet-lens-candidates", dest="drop_wet_lens_candidates", action="store_false", help="Review wet-lens candidates instead of dropping them")
    ap.add_argument("--keep-fog-haze-candidates", dest="drop_fog_haze_candidates", action="store_false", help="Review fog/haze candidates instead of dropping them")
    ap.add_argument("--keep-obscured-candidates", dest="drop_obscured_candidates", action="store_false", help="Review obscured/washed-out candidates instead of dropping them")
    ap.add_argument("--keep-glare-candidates", dest="drop_glare_candidates", action="store_false", help="Review strong glare/overexposure candidates instead of dropping them")
    ap.add_argument("--keep-dark-candidates", dest="drop_dark_candidates", action="store_false", help="Review dark candidates instead of dropping them")
    ap.add_argument("--keep-blurry-candidates", dest="drop_blurry_candidates", action="store_false", help="Review blurry candidates instead of dropping them")
    ap.add_argument("--keep-snow-ice-candidates", dest="drop_snow_ice_candidates", action="store_false", help="Review snow/ice candidates instead of dropping them")
    ap.add_argument("--fog-gray-std-ceiling", type=float, default=32.0, help="Base contrast ceiling for fog/haze filtering")
    ap.add_argument("--fog-colorfulness-ceiling", type=float, default=14.0, help="Colorfulness ceiling for fog/haze filtering")
    ap.add_argument("--wet-lens-min-blob-count", type=int, default=6, help="Minimum bright low-saturation blobs for wet-lens filtering")
    ap.add_argument("--obscured-bright-low-sat-ratio", type=float, default=0.10, help="Bright low-saturation area ratio for obscured/washed-out filtering")
    ap.add_argument("--glare-very-bright-low-sat-ratio", type=float, default=0.16, help="Very bright low-saturation area ratio for glare filtering")
    ap.add_argument("--glare-saturated-ratio", type=float, default=0.15, help="Near-saturated bright pixel ratio for glare filtering")
    ap.add_argument("--glare-bright-low-sat-ratio", type=float, default=0.16, help="Bright low-saturation support ratio for glare filtering")
    ap.add_argument("--glare-peak-v-threshold", type=float, default=0.88, help="Brightness histogram peak needed for broad glare filtering")
    ap.add_argument("--snow-white-low-sat-ratio", type=float, default=0.30, help="White low-saturation area ratio for snow/ice filtering")
    ap.add_argument("--snow-large-white-low-sat-ratio", type=float, default=0.25, help="Large white low-saturation area ratio for snow/ice filtering")
    ap.add_argument("--snow-strong-white-low-sat-ratio", type=float, default=0.50, help="Strong white low-saturation area ratio for snow/ice filtering")
    ap.add_argument("--snow-cool-v-thresh", type=float, default=0.45, help="Brightness floor for broad blue/cool snow filtering")
    ap.add_argument("--snow-cool-s-thresh", type=float, default=0.55, help="Saturation ceiling for broad blue/cool snow filtering")
    ap.add_argument("--snow-cool-bright-ratio", type=float, default=0.55, help="Broad blue/cool bright area ratio for snow/ice filtering")
    ap.add_argument("--snow-blue-v-thresh", type=float, default=0.35, help="Brightness floor for saturated blue snow filtering")
    ap.add_argument("--snow-blue-s-floor", type=float, default=0.30, help="Saturation floor for saturated blue snow filtering")
    ap.add_argument("--snow-blue-margin", type=float, default=0.12, help="Minimum normalized blue-minus-red margin for blue snow filtering")
    ap.add_argument("--snow-blue-dominant-ratio", type=float, default=0.85, help="Broad saturated blue area ratio for snow/ice filtering")
    ap.add_argument("--snow-blue-gray-mean-floor", type=float, default=80.0, help="Mean brightness floor for saturated blue snow filtering")
    ap.add_argument("--snow-blue-gray-std-ceiling", type=float, default=40.0, help="Contrast ceiling for saturated blue snow filtering")
    ap.add_argument("--snow-gray-mean-floor", type=float, default=105.0, help="Mean brightness floor for snow/ice filtering")
    ap.add_argument("--snow-cool-gray-mean-floor", type=float, default=95.0, help="Mean brightness floor for blue/cool snow filtering")
    ap.add_argument("--snow-cool-gray-std-ceiling", type=float, default=42.0, help="Contrast ceiling for blue/cool snow filtering")
    ap.add_argument("--snow-colorfulness-ceiling", type=float, default=22.0, help="Colorfulness ceiling for snow/ice filtering")
    ap.add_argument("--ice-gray-low-sat-ratio", type=float, default=0.55, help="Large gray low-saturation area ratio for ice/frozen-water filtering")
    ap.add_argument("--ice-gray-mean-floor", type=float, default=95.0, help="Mean brightness floor for gray ice filtering")
    ap.add_argument("--ice-gray-std-ceiling", type=float, default=38.0, help="Contrast ceiling for gray ice filtering")
    ap.add_argument("--ice-colorfulness-ceiling", type=float, default=35.0, help="Colorfulness ceiling for gray ice filtering")
    ap.add_argument("--snow-months", default="11,12,1,2,3,4", help="Comma-separated local months where snow/ice filtering is enabled")
    ap.add_argument("--max-soft-flags-before-drop", type=int, default=3, help="Drop image if it accumulates at least this many soft flags")
    args = ap.parse_args()

    cfg = QualityConfig(
        test_water_year=args.test_water_year,
        fallback_val_ratio=args.fallback_val_ratio,
        min_images_per_water_year=args.min_images_per_water_year,
        use_local_daytime_filter=not args.disable_local_daytime_filter,
        use_site_day_windows=not args.disable_site_day_windows,
        local_day_start_hour=args.local_day_start_hour,
        local_day_end_hour=args.local_day_end_hour,
        night_grayscale_score=args.night_grayscale_score,
        night_dark_gray_mean=args.night_dark_gray_mean,
        night_dark_peak_v=args.night_dark_peak_v,
        dark_gray_mean_floor=args.dark_gray_mean_floor,
        severe_dark_gray_mean=args.severe_dark_gray_mean,
        low_contrast_gray_std_floor=args.low_contrast_gray_std_floor,
        severe_low_contrast_gray_std=args.severe_low_contrast_gray_std,
        blur_lap_var_floor=args.blur_lap_var_floor,
        severe_blur_lap_var=args.severe_blur_lap_var,
        blur_quantile=args.blur_quantile,
        blur_threshold_multiplier=args.blur_threshold_multiplier,
        low_colorfulness_floor=args.low_colorfulness_floor,
        severe_low_colorfulness=args.severe_low_colorfulness,
        drop_rain_candidates=args.drop_rain_candidates,
        drop_wet_lens_candidates=args.drop_wet_lens_candidates,
        drop_fog_haze_candidates=args.drop_fog_haze_candidates,
        drop_obscured_candidates=args.drop_obscured_candidates,
        drop_glare_candidates=args.drop_glare_candidates,
        drop_dark_candidates=args.drop_dark_candidates,
        drop_blurry_candidates=args.drop_blurry_candidates,
        drop_snow_ice_candidates=args.drop_snow_ice_candidates,
        fog_gray_std_ceiling=args.fog_gray_std_ceiling,
        fog_colorfulness_ceiling=args.fog_colorfulness_ceiling,
        wet_lens_min_blob_count=args.wet_lens_min_blob_count,
        obscured_bright_low_sat_ratio=args.obscured_bright_low_sat_ratio,
        glare_very_bright_low_sat_ratio=args.glare_very_bright_low_sat_ratio,
        glare_saturated_ratio=args.glare_saturated_ratio,
        glare_bright_low_sat_ratio=args.glare_bright_low_sat_ratio,
        glare_peak_v_threshold=args.glare_peak_v_threshold,
        snow_white_low_sat_ratio=args.snow_white_low_sat_ratio,
        snow_large_white_low_sat_ratio=args.snow_large_white_low_sat_ratio,
        snow_strong_white_low_sat_ratio=args.snow_strong_white_low_sat_ratio,
        snow_cool_v_thresh=args.snow_cool_v_thresh,
        snow_cool_s_thresh=args.snow_cool_s_thresh,
        snow_cool_bright_ratio=args.snow_cool_bright_ratio,
        snow_blue_v_thresh=args.snow_blue_v_thresh,
        snow_blue_s_floor=args.snow_blue_s_floor,
        snow_blue_margin=args.snow_blue_margin,
        snow_blue_dominant_ratio=args.snow_blue_dominant_ratio,
        snow_blue_gray_mean_floor=args.snow_blue_gray_mean_floor,
        snow_blue_gray_std_ceiling=args.snow_blue_gray_std_ceiling,
        snow_gray_mean_floor=args.snow_gray_mean_floor,
        snow_cool_gray_mean_floor=args.snow_cool_gray_mean_floor,
        snow_cool_gray_std_ceiling=args.snow_cool_gray_std_ceiling,
        snow_colorfulness_ceiling=args.snow_colorfulness_ceiling,
        ice_gray_low_sat_ratio=args.ice_gray_low_sat_ratio,
        ice_gray_mean_floor=args.ice_gray_mean_floor,
        ice_gray_std_ceiling=args.ice_gray_std_ceiling,
        ice_colorfulness_ceiling=args.ice_colorfulness_ceiling,
        snow_months=args.snow_months,
        max_soft_flags_before_drop=args.max_soft_flags_before_drop,
    )

    output_root = Path(args.output_root).resolve()
    site_info_csv = Path(args.site_info_csv).resolve() if args.site_info_csv else None
    sites = choose_sites(output_root=output_root, site_info_csv=site_info_csv, site_ids=args.site_id)
    if not sites:
        raise SystemExit("No sites selected for filtering.")

    summaries = []
    for site_id, site_dir in sites:
        print(f"[filter] {site_id} -> {site_dir}", flush=True)
        summaries.append(
            filter_single_site(
                site_id=site_id,
                site_dir=site_dir,
                cfg=cfg,
                activate=args.activate_filtered,
                filtered_dir_name=args.filtered_dir_name,
                show_progress=not args.no_progress,
            )
        )

    out = {
        "n_sites": len(summaries),
        "activated": bool(args.activate_filtered),
        "sites": summaries,
        "config": asdict(cfg),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
