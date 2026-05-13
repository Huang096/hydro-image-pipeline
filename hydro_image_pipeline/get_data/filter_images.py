from __future__ import annotations

import argparse
import zlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .filter_site_images import (
        QualityConfig,
        SITE_DAY_WINDOWS,
        SITE_TIMEZONES,
        classify_quality,
        hardlink_or_copy,
        load_quality_metrics,
        progress_iter,
        strip_previous_quality_columns,
        summarize_water_years,
    )
except ImportError:  # Allows direct execution as a script.
    from filter_site_images import (
        QualityConfig,
        SITE_DAY_WINDOWS,
        SITE_TIMEZONES,
        classify_quality,
        hardlink_or_copy,
        load_quality_metrics,
        progress_iter,
        strip_previous_quality_columns,
        summarize_water_years,
    )


CENTRAL_GROUPS = {"WI_Chippewa_River", "WI_Milwaukee_River", "MN_Redwood_River"}
MOUNTAIN_GROUPS = {"ND_Little_Missouri_River"}
PIPELINE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PIPELINE_ROOT / "data"


def water_year(ts: pd.Series) -> pd.Series:
    return ts.dt.year + (ts.dt.month >= 10).astype(int)


def water_year_season(month: pd.Series) -> pd.Series:
    m = month.astype(int)
    return pd.Series(
        np.select(
            [
                m.isin([10, 11, 12]),
                m.isin([1, 2, 3]),
                m.isin([4, 5, 6]),
                m.isin([7, 8, 9]),
            ],
            ["fall", "winter", "spring", "summer"],
            default="unknown",
        ),
        index=month.index,
    )


def site_timezone(river_group: str) -> str:
    if river_group in CENTRAL_GROUPS:
        return "America/Chicago"
    if river_group in MOUNTAIN_GROUPS:
        return "America/Denver"
    return "America/New_York"


def register_site_time_settings(site_name: str, river_group: str, day_start: float, day_end: float) -> None:
    SITE_TIMEZONES[site_name] = site_timezone(river_group)
    SITE_DAY_WINDOWS[site_name] = (float(day_start), float(day_end))


def assign_site_splits(
    kept: pd.DataFrame,
    site_name: str,
    test_water_year: int,
    fallback_val_ratio: float,
    min_images_per_water_year: int,
    rough_year_min_months: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    df = kept.copy()
    df["image_time"] = pd.to_datetime(df["image_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["image_time"]).sort_values("image_time").reset_index(drop=True)
    df["water_year"] = water_year(df["image_time"])
    df["month_in_wy"] = ((df["image_time"].dt.month + 2) % 12) + 1
    df["season"] = water_year_season(df["image_time"].dt.month)

    wy_counts = df["water_year"].value_counts().sort_index()
    wy_month_counts = df.groupby("water_year")["month_in_wy"].nunique().sort_index()
    eligible_wys = [int(wy) for wy, n in wy_counts.items() if int(n) >= int(min_images_per_water_year)]
    if test_water_year not in eligible_wys:
        raise ValueError(f"WY{test_water_year} has too few kept images after filtering; counts={wy_counts.to_dict()}")

    pre_test_wys = [wy for wy in eligible_wys if wy < test_water_year]
    if not pre_test_wys:
        raise ValueError(f"no pre-WY{test_water_year} images left for train/val; counts={wy_counts.to_dict()}")

    use_wys = pre_test_wys + [int(test_water_year)]
    df = df[df["water_year"].isin(use_wys)].copy()
    df["split"] = "unused"
    df.loc[df["water_year"] == int(test_water_year), "split"] = "test"

    split_meta: dict[str, object] = {
        "test_water_year": int(test_water_year),
        "all_water_year_counts_after_quality_filter": {str(int(k)): int(v) for k, v in wy_counts.items()},
        "all_water_year_month_counts_after_quality_filter": {
            str(int(k)): int(v) for k, v in wy_month_counts.items()
        },
        "used_water_years": [int(x) for x in use_wys],
        "excluded_water_years": [int(x) for x in eligible_wys if x not in use_wys],
        "rough_year_min_months": int(rough_year_min_months),
    }

    rough_pre_test_wys = [
        wy
        for wy in pre_test_wys
        if int(wy_month_counts.get(wy, 0)) >= int(rough_year_min_months)
        and int(wy_counts.get(wy, 0)) >= int(min_images_per_water_year)
    ]
    if len(rough_pre_test_wys) >= 2:
        val_wy = rough_pre_test_wys[-1]
        train_wys = rough_pre_test_wys[:-1]
        df.loc[df["water_year"].isin(train_wys), "split"] = "train"
        df.loc[df["water_year"] == val_wy, "split"] = "val"
        split_meta.update(
            {
                "split_strategy": "rough_full_water_years",
                "train_water_years": [int(x) for x in train_wys],
                "val_water_year": int(val_wy),
                "ignored_incomplete_pre_test_water_years": [int(x) for x in pre_test_wys if x not in rough_pre_test_wys],
            }
        )
    else:
        pre_test_chunk = df[df["water_year"].isin(pre_test_wys)].sort_values("image_time")
        rng_seed = int(seed) + int(zlib.crc32(site_name.encode("utf-8")) & 0xFFFFFFFF)
        val_indices: list[int] = []
        val_by_season: dict[str, int] = {}
        for i, season in enumerate(["fall", "winter", "spring", "summer"]):
            season_idx = pre_test_chunk.index[pre_test_chunk["season"] == season]
            n = int(len(season_idx))
            if n <= 1:
                val_n = 0
            else:
                val_n = min(n - 1, max(1, int(round(n * float(fallback_val_ratio)))))
            if val_n:
                sampled = pd.Series(season_idx).sample(n=val_n, random_state=rng_seed + i).tolist()
                val_indices.extend(sampled)
            val_by_season[season] = int(val_n)

        df.loc[pre_test_chunk.index, "split"] = "train"
        df.loc[val_indices, "split"] = "val"
        split_meta.update(
            {
                "split_strategy": "season_stratified_pre_test_pool",
                "train_water_years": [int(x) for x in pre_test_wys],
                "val_water_years": [int(x) for x in pre_test_wys],
                "fallback_val_ratio": float(fallback_val_ratio),
                "rough_pre_test_water_years_available": [int(x) for x in rough_pre_test_wys],
                "val_by_season": val_by_season,
            }
        )

    counts = df["split"].value_counts().to_dict()
    for split in ("train", "val", "test"):
        if int(counts.get(split, 0)) == 0:
            raise ValueError(f"split={split} is empty after assignment; counts={counts}")
    split_meta["split_counts"] = {str(k): int(v) for k, v in counts.items()}
    return df.drop(columns=["month_in_wy", "season"]).sort_values(["split", "image_time"]).reset_index(drop=True), split_meta


def reset_split_dirs(site_dir: Path) -> None:
    for split in ("train", "val", "test"):
        split_dir = site_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)


def materialize_split_dirs(split_df: pd.DataFrame, site_dir: Path, show_progress: bool) -> None:
    reset_split_dirs(site_dir)
    for row in progress_iter(
        split_df.to_dict(orient="records"),
        total=len(split_df),
        label=f"{site_dir.name} build split dirs",
        enabled=show_progress,
    ):
        src = Path(str(row["image_path"])).resolve()
        dst = site_dir / str(row["split"]) / src.name
        hardlink_or_copy(src, dst)


def backup_once(path: Path, suffix: str = ".pre_image_filter_backup") -> None:
    backup = path.with_suffix(path.suffix + suffix)
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)


def filter_one_site(
    site_dir: Path,
    river_group: str,
    cfg: QualityConfig,
    day_start: float,
    day_end: float,
    rough_year_min_months: int,
    seed: int,
    activate: bool,
    show_progress: bool,
) -> dict[str, object]:
    site_name = site_dir.name
    register_site_time_settings(site_name, river_group, day_start, day_end)

    matched_path = site_dir / "matched.csv"
    if not matched_path.exists():
        raise FileNotFoundError(f"missing matched.csv: {matched_path}")

    matched = strip_previous_quality_columns(pd.read_csv(matched_path))
    if "image_path" not in matched.columns or "image_time" not in matched.columns:
        raise ValueError(f"{matched_path} must contain image_path and image_time")

    metrics = pd.DataFrame(
        [
            load_quality_metrics(Path(p), cfg)
            for p in progress_iter(
                matched["image_path"].astype(str).tolist(),
                total=len(matched),
                label=f"{site_name} quality scan",
                enabled=show_progress,
            )
        ]
    )
    metadata = matched[["image_path", "image_time"]].rename(columns={"image_time": "quality_image_time"})
    metadata["quality_site_id"] = site_name
    metrics = metrics.merge(metadata, on="image_path", how="left", validate="one_to_one")
    classified, thresholds = classify_quality(metrics, cfg)
    audit = matched.merge(classified, on="image_path", how="left", validate="one_to_one")
    audit_path = site_dir / "image_quality_audit.csv"
    audit.to_csv(audit_path, index=False)

    kept = audit[audit["quality_keep"].fillna(False)].copy()
    if kept.empty:
        raise ValueError(f"all images were filtered out: {site_dir}")

    original_cols = matched.columns.tolist()
    kept_records = kept[original_cols].copy()
    matched_filtered_path = site_dir / "matched_filtered.csv"
    kept_records.to_csv(matched_filtered_path, index=False)

    wy_summary = summarize_water_years(kept_records, cfg.min_images_per_water_year)
    wy_summary_path = site_dir / "water_year_summary_filtered.csv"
    wy_summary.to_csv(wy_summary_path, index=False)

    split_df, split_meta = assign_site_splits(
        kept=kept_records,
        site_name=site_name,
        test_water_year=cfg.test_water_year,
        fallback_val_ratio=cfg.fallback_val_ratio,
        min_images_per_water_year=cfg.min_images_per_water_year,
        rough_year_min_months=rough_year_min_months,
        seed=seed,
    )
    split_path = site_dir / "split_manifest_filtered.csv"
    split_df.to_csv(split_path, index=False)

    if activate:
        backup_once(site_dir / "matched.csv")
        backup_once(site_dir / "split_manifest.csv")
        backup_once(site_dir / "water_year_summary.csv")
        shutil.copy2(matched_filtered_path, site_dir / "matched.csv")
        shutil.copy2(split_path, site_dir / "split_manifest.csv")
        shutil.copy2(wy_summary_path, site_dir / "water_year_summary.csv")
        materialize_split_dirs(split_df, site_dir=site_dir, show_progress=show_progress)

    dropped = audit.loc[~audit["quality_keep"].fillna(False), "quality_reasons"].value_counts().to_dict()
    summary = {
        "river_group": river_group,
        "site_name": site_name,
        "site_dir": str(site_dir.resolve()),
        "timezone": SITE_TIMEZONES[site_name],
        "day_window": {"start": float(day_start), "end": float(day_end)},
        "n_input_rows": int(len(matched)),
        "n_kept_rows": int(len(kept)),
        "n_dropped_rows": int(len(matched) - len(kept)),
        "keep_ratio": float(len(kept) / max(1, len(matched))),
        "drop_reason_counts": {str(k): int(v) for k, v in dropped.items()},
        "review_count": int(audit["quality_review"].fillna(False).sum()),
        "quality_thresholds": thresholds,
        "split_meta": split_meta,
        "activated": bool(activate),
        "files": {
            "audit_csv": str(audit_path.resolve()),
            "matched_filtered_csv": str(matched_filtered_path.resolve()),
            "water_year_summary_filtered_csv": str(wy_summary_path.resolve()),
            "split_manifest_filtered_csv": str(split_path.resolve()),
        },
    }
    (site_dir / "filter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def discover_site_dirs(output_root: Path) -> list[tuple[str, Path]]:
    records = []
    for river_dir in sorted(output_root.iterdir()):
        if not river_dir.is_dir():
            continue
        for site_dir in sorted(river_dir.iterdir()):
            if site_dir.is_dir() and (site_dir / "matched.csv").exists():
                records.append((river_dir.name, site_dir))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality-filter downloaded images and build train/val/test splits.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--test-water-year", type=int, default=2025)
    parser.add_argument("--fallback-val-ratio", type=float, default=0.2)
    parser.add_argument("--min-images-per-water-year", type=int, default=1)
    parser.add_argument("--day-start", type=float, default=8.0)
    parser.add_argument("--day-end", type=float, default=18.5)
    parser.add_argument("--rough-year-min-months", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--site-name", action="append", default=[], help="Optional site folder name; can repeat.")
    parser.add_argument("--activate-filtered", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    cfg = QualityConfig(
        test_water_year=args.test_water_year,
        fallback_val_ratio=args.fallback_val_ratio,
        min_images_per_water_year=args.min_images_per_water_year,
    )
    root = args.output_root.resolve()
    sites = discover_site_dirs(root)
    if args.site_name:
        wanted = set(args.site_name)
        sites = [(river, path) for river, path in sites if path.name in wanted]
    if not sites:
        raise SystemExit(f"no site folders with matched.csv found under {root}")

    summaries = []
    for idx, (river_group, site_dir) in enumerate(sites, start=1):
        print(f"[{idx}/{len(sites)}] filter {river_group} / {site_dir.name}", flush=True)
        summaries.append(
            filter_one_site(
                site_dir=site_dir,
                river_group=river_group,
                cfg=cfg,
                day_start=args.day_start,
                day_end=args.day_end,
                rough_year_min_months=args.rough_year_min_months,
                seed=args.seed,
                activate=args.activate_filtered,
                show_progress=not args.no_progress,
            )
        )

    out = {
        "output_root": str(root),
        "n_sites": int(len(summaries)),
        "activated": bool(args.activate_filtered),
        "config": asdict(cfg),
        "sites": summaries,
    }
    out_path = root / "image_filter_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "river_group": item["river_group"],
                "site_name": item["site_name"],
                "n_input_rows": item["n_input_rows"],
                "n_kept_rows": item["n_kept_rows"],
                "n_dropped_rows": item["n_dropped_rows"],
                "keep_ratio": item["keep_ratio"],
                "split_counts": json.dumps(item["split_meta"]["split_counts"], sort_keys=True),
                "split_strategy": item["split_meta"]["split_strategy"],
            }
            for item in summaries
        ]
    ).to_csv(root / "image_filter_summary.csv", index=False)
    print(json.dumps({"summary_json": str(out_path), "n_sites": len(summaries)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
