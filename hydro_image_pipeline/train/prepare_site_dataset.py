from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})")


def parse_image_time(value: str) -> pd.Timestamp:
    match = TIMESTAMP_RE.search(value)
    if not match:
        raise ValueError(f"cannot parse timestamp from image name: {value}")
    return pd.to_datetime(match.group(1), format="%Y-%m-%dT%H-%M-%S", utc=True)


def discover_images(site_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(site_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            ts = parse_image_time(path.name)
        except ValueError:
            continue
        rows.append(
            {
                "filename": path.name,
                "image_path": str(path.resolve()),
                "image_time_dt": ts,
                "image_time": ts.isoformat(),
                "year": int(ts.year),
                "month": int(ts.month),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no timestamped images found under: {site_dir}")
    return pd.DataFrame(rows).sort_values("image_time_dt").reset_index(drop=True)


def discover_images_from_split_dirs(site_dir: Path, split_dirs: dict[str, str], roi_path: Path | None = None) -> pd.DataFrame:
    rows = []
    resolved_roi_path = (roi_path if roi_path is not None else site_dir / "roi.json").resolve()
    for split_name, rel_dir in split_dirs.items():
        split_dir = site_dir / rel_dir
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                ts = parse_image_time(path.name)
            except ValueError:
                continue
            rows.append(
                {
                    "filename": path.name,
                    "image_path": str(path.resolve()),
                    "image_time_dt": ts,
                    "image_time": ts.isoformat(),
                    "year": int(ts.year),
                    "month": int(ts.month),
                    "split": split_name,
                    "site_dir": str(site_dir.resolve()),
                    "roi_path": str(resolved_roi_path),
                }
            )
    if not rows:
        raise FileNotFoundError(f"no timestamped images found under split directories of: {site_dir}")
    return pd.DataFrame(rows).sort_values(["split", "image_time_dt"]).reset_index(drop=True)


def _choose_time_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_origin_table(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["variable_cd"] = tmp["variable_cd"].astype(str)
    tmp["value"] = pd.to_numeric(tmp["value"], errors="coerce")
    time_col = _choose_time_column(tmp, ["datetime_utc", "time", "hydro_time"])
    if time_col is None:
        raise ValueError("origin-style table is missing a time column")
    pivot = (
        tmp.pivot_table(index=time_col, columns="variable_cd", values="value", aggfunc="last")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    rename_map = {"60": "discharge", "00060": "discharge", "65": "gauge_height", "00065": "gauge_height"}
    pivot = pivot.rename(columns=rename_map)
    return pivot


def load_labels_table(labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    if {"variable_cd", "value"} <= set(df.columns):
        df = _normalize_origin_table(df)
    return df


def _standardize_direct_labels(df: pd.DataFrame) -> pd.DataFrame | None:
    cols = set(df.columns)
    if "discharge" not in cols:
        return None

    tmp = df.copy()
    if "filename" not in tmp.columns:
        if "image_path" in tmp.columns:
            tmp["filename"] = tmp["image_path"].map(lambda x: Path(str(x)).name)
        elif "path" in tmp.columns:
            tmp["filename"] = tmp["path"].map(lambda x: Path(str(x)).name)

    if "image_time" not in tmp.columns:
        for col in ["time_img_utc", "hydro_time", "time", "datetime_utc"]:
            if col in tmp.columns:
                tmp["image_time"] = tmp[col]
                break

    if "hydro_time" not in tmp.columns and "image_time" in tmp.columns:
        tmp["hydro_time"] = tmp["image_time"]

    required = {"filename", "image_time", "hydro_time", "discharge"}
    if not required <= set(tmp.columns):
        return None

    if "gauge_height" not in tmp.columns:
        tmp["gauge_height"] = pd.NA

    tmp["image_time_dt"] = pd.to_datetime(tmp["image_time"], utc=True, errors="coerce")
    tmp["hydro_time_dt"] = pd.to_datetime(tmp["hydro_time"], utc=True, errors="coerce")
    tmp["discharge"] = pd.to_numeric(tmp["discharge"], errors="coerce")
    tmp["gauge_height"] = pd.to_numeric(tmp["gauge_height"], errors="coerce")
    tmp = tmp.dropna(subset=["filename", "image_time_dt", "hydro_time_dt", "discharge"]).copy()
    return tmp[
        ["filename", "hydro_time_dt", "discharge", "gauge_height"]
    ].drop_duplicates(subset=["filename"])


def _standardize_hydro_table(df: pd.DataFrame) -> pd.DataFrame | None:
    cols = set(df.columns)
    if "discharge" not in cols:
        return None
    time_col = _choose_time_column(df, ["hydro_time", "time", "datetime_utc", "image_time"])
    if time_col is None:
        return None

    tmp = df.copy()
    tmp["hydro_time_dt"] = pd.to_datetime(tmp[time_col], utc=True, errors="coerce")
    tmp["discharge"] = pd.to_numeric(tmp["discharge"], errors="coerce")
    if "gauge_height" not in tmp.columns:
        tmp["gauge_height"] = pd.NA
    tmp["gauge_height"] = pd.to_numeric(tmp["gauge_height"], errors="coerce")
    tmp = tmp.dropna(subset=["hydro_time_dt", "discharge"]).copy()
    return tmp[["hydro_time_dt", "discharge", "gauge_height"]].sort_values("hydro_time_dt").reset_index(drop=True)


def merge_labels(
    images: pd.DataFrame,
    labels_df: pd.DataFrame,
    match_tolerance_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    direct = _standardize_direct_labels(labels_df)
    if direct is not None:
        merged = images.merge(direct, on="filename", how="left")
    else:
        hydro = _standardize_hydro_table(labels_df)
        if hydro is None:
            raise ValueError(
                "labels CSV is not recognized. Expected either per-image labels "
                "(filename/image_path + discharge) or a hydro table (time + discharge)."
            )
        merged = pd.merge_asof(
            images.sort_values("image_time_dt"),
            hydro,
            left_on="image_time_dt",
            right_on="hydro_time_dt",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=match_tolerance_minutes),
        )

    unmatched = merged[merged["discharge"].isna()].copy()
    matched = merged[merged["discharge"].notna()].copy()
    matched["hydro_time_dt"] = pd.to_datetime(matched["hydro_time_dt"], utc=True, errors="coerce")
    matched["hydro_time"] = matched["hydro_time_dt"].map(lambda x: x.isoformat() if pd.notna(x) else "")
    matched["image_time"] = matched["image_time_dt"].map(lambda x: x.isoformat())
    matched["year"] = matched["image_time_dt"].dt.year.astype(int)
    matched["month"] = matched["image_time_dt"].dt.month.astype(int)
    keep_cols = ["image_path", "image_time", "year", "month", "hydro_time", "discharge", "gauge_height"]
    if "split" in matched.columns:
        keep_cols = ["split"] + keep_cols
    if "site_id" in matched.columns:
        keep_cols = ["site_id"] + keep_cols
    if "site_dir" in matched.columns:
        keep_cols = ["site_dir"] + keep_cols
    if "roi_path" in matched.columns:
        keep_cols = ["roi_path"] + keep_cols
    matched = matched[keep_cols].sort_values([c for c in ["site_id", "split", "image_time"] if c in keep_cols]).reset_index(drop=True)
    return matched, unmatched


def auto_detect_labels_csv(site_dir: Path) -> Path | None:
    candidates = [
        "matched.csv",
        "labels.csv",
        "images_labeled_exact.csv",
        "features_raw.csv",
        "origin.csv",
        "long_table.csv",
    ]
    for name in candidates:
        path = site_dir / name
        if path.exists():
            return path
    return None


def build_site_dataset(site_dir: Path, labels_csv: Path, match_tolerance_minutes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    images = discover_images(site_dir)
    labels_df = load_labels_table(labels_csv)
    return merge_labels(images=images, labels_df=labels_df, match_tolerance_minutes=match_tolerance_minutes)


def build_site_dataset_from_splits(
    site_dir: Path,
    labels_csv: Path,
    match_tolerance_minutes: int,
    split_dirs: dict[str, str],
    site_id: str | None = None,
    roi_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved_roi_path = (roi_path if roi_path is not None else site_dir / "roi.json").resolve()
    images = discover_images_from_split_dirs(site_dir=site_dir, split_dirs=split_dirs, roi_path=resolved_roi_path)
    if site_id:
        images["site_id"] = site_id
    labels_df = load_labels_table(labels_csv)
    matched, unmatched = merge_labels(images=images, labels_df=labels_df, match_tolerance_minutes=match_tolerance_minutes)
    if site_id:
        matched["site_id"] = site_id
        unmatched["site_id"] = site_id
    matched["site_dir"] = str(site_dir.resolve())
    matched["roi_path"] = str(resolved_roi_path)
    unmatched["site_dir"] = str(site_dir.resolve())
    unmatched["roi_path"] = str(resolved_roi_path)
    return matched, unmatched


def load_site_list(site_list_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(site_list_csv, dtype={"site_id": "string"})
    required = {"site_id", "site_dir"}
    if not required <= set(df.columns):
        raise ValueError(f"site list must contain columns: {sorted(required)}")
    return df


def build_multi_site_dataset(
    site_list_csv: Path,
    match_tolerance_minutes: int,
    split_dirs: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    site_df = load_site_list(site_list_csv)
    matched_parts = []
    unmatched_parts = []
    for row in site_df.to_dict(orient="records"):
        site_id = str(row["site_id"])
        site_dir = Path(str(row["site_dir"])).resolve()
        labels_csv_raw = str(row.get("labels_csv", "")).strip()
        labels_csv = Path(labels_csv_raw).resolve() if labels_csv_raw else auto_detect_labels_csv(site_dir)
        roi_path_raw = str(row.get("roi_path", "")).strip()
        roi_path = Path(roi_path_raw).resolve() if roi_path_raw else None
        if labels_csv is None:
            raise FileNotFoundError(f"could not auto-detect labels CSV for site_id={site_id} site_dir={site_dir}")
        matched, unmatched = build_site_dataset_from_splits(
            site_dir=site_dir,
            labels_csv=labels_csv,
            match_tolerance_minutes=match_tolerance_minutes,
            split_dirs=split_dirs,
            site_id=site_id,
            roi_path=roi_path,
        )
        matched_parts.append(matched)
        unmatched_parts.append(unmatched)
    matched_all = pd.concat(matched_parts, ignore_index=True) if matched_parts else pd.DataFrame()
    unmatched_all = pd.concat(unmatched_parts, ignore_index=True) if unmatched_parts else pd.DataFrame()
    return matched_all, unmatched_all


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a standard site manifest for the best 0331 pipeline.")
    ap.add_argument("--site-dir", default="", help="Single site folder containing labels and split image folders.")
    ap.add_argument("--labels-csv", default="", help="Label table. If omitted, try to auto-detect inside site-dir.")
    ap.add_argument("--output-csv", required=True, help="Where to write the standardized manifest CSV.")
    ap.add_argument("--site-list-csv", default="", help="CSV containing site_id, site_dir, and optional labels_csv.")
    ap.add_argument(
        "--unmatched-csv",
        default="",
        help="Optional CSV path for images that could not be matched to discharge labels.",
    )
    ap.add_argument(
        "--match-tolerance-minutes",
        type=int,
        default=30,
        help="Nearest-time tolerance used when labels CSV is a hydro time series.",
    )
    ap.add_argument("--train-dir-name", default="train", help="Subfolder name for training images inside each site.")
    ap.add_argument("--val-dir-name", default="val", help="Subfolder name for validation images inside each site.")
    ap.add_argument("--test-dir-name", default="test", help="Subfolder name for test images inside each site.")
    args = ap.parse_args()

    split_dirs = {"train": args.train_dir_name, "val": args.val_dir_name, "test": args.test_dir_name}
    if not args.site_dir and not args.site_list_csv:
        raise ValueError("please provide either --site-dir or --site-list-csv")
    if args.site_list_csv:
        site_list_csv = Path(args.site_list_csv).resolve()
        matched, unmatched = build_multi_site_dataset(
            site_list_csv=site_list_csv,
            match_tolerance_minutes=args.match_tolerance_minutes,
            split_dirs=split_dirs,
        )
        source_summary = {"site_list_csv": str(site_list_csv)}
    else:
        site_dir = Path(args.site_dir).resolve()
        labels_csv = Path(args.labels_csv).resolve() if args.labels_csv else auto_detect_labels_csv(site_dir)
        if labels_csv is None:
            raise FileNotFoundError("could not auto-detect a labels CSV inside site-dir")
        matched, unmatched = build_site_dataset_from_splits(
            site_dir=site_dir,
            labels_csv=labels_csv,
            match_tolerance_minutes=args.match_tolerance_minutes,
            split_dirs=split_dirs,
            site_id=site_dir.name,
        )
        source_summary = {"site_dir": str(site_dir), "labels_csv": str(labels_csv)}

    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output_csv, index=False)

    if args.unmatched_csv:
        unmatched_csv = Path(args.unmatched_csv).resolve()
        unmatched_csv.parent.mkdir(parents=True, exist_ok=True)
        unmatched.to_csv(unmatched_csv, index=False)

    summary = {
        "n_images_total": int(len(matched) + len(unmatched)),
        "n_images_matched": int(len(matched)),
        "n_images_unmatched": int(len(unmatched)),
        "split_counts": matched["split"].value_counts().to_dict() if "split" in matched.columns and len(matched) else {},
        "output_csv": str(output_csv),
    }
    summary.update(source_summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
