from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

try:
    from .prepare_site_dataset import auto_detect_labels_csv, build_multi_site_dataset, build_site_dataset_from_splits
except ImportError:  # Allows direct execution as a script.
    from prepare_site_dataset import auto_detect_labels_csv, build_multi_site_dataset, build_site_dataset_from_splits

BUNDLE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BUNDLE_DIR / "best_pipeline_config.json"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_best_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_feature_table(
    meta: pd.DataFrame,
    roi_path: Path | None,
    backbone: str,
    batch_size: int,
    refine_water_mask: bool,
) -> pd.DataFrame:
    extract_mod = load_module("pipeline_extract_mod", BUNDLE_DIR / "extract_torchvision_backbone_features_v7.py")
    if roi_path is not None:
        roi = extract_mod.load_roi(roi_path)
        arr, _ = extract_mod.extract(
            meta=meta,
            roi=roi,
            backbone=backbone,
            batch_size=batch_size,
            progress_label="features all-sites",
            refine_water_mask=refine_water_mask,
        )
        emb_df = pd.DataFrame(arr, columns=[f"emb_{i:03d}" for i in range(arr.shape[1])])
        return pd.concat([meta.reset_index(drop=True), emb_df], axis=1)

    if "roi_path" not in meta.columns:
        raise ValueError("feature extraction needs roi_path column when no global roi_path override is provided")

    parts = []
    groups = list(meta.groupby("roi_path", sort=False))
    for group_idx, (roi_value, chunk) in enumerate(groups, start=1):
        roi = extract_mod.load_roi(Path(str(roi_value)))
        if "site_id" in chunk.columns:
            site_label = ",".join(sorted(chunk["site_id"].astype(str).unique().tolist()))
        else:
            site_label = Path(str(roi_value)).parent.name
        print(
            f"[features] ROI group {group_idx}/{len(groups)} | site={site_label} "
            f"| rows={len(chunk)} | roi={roi_value}",
            flush=True,
        )
        arr, _ = extract_mod.extract(
            meta=chunk.reset_index(drop=True),
            roi=roi,
            backbone=backbone,
            batch_size=batch_size,
            progress_label=f"features {group_idx}/{len(groups)} {site_label}",
            refine_water_mask=refine_water_mask,
        )
        emb_df = pd.DataFrame(arr, columns=[f"emb_{i:03d}" for i in range(arr.shape[1])])
        parts.append(pd.concat([chunk.reset_index(drop=True), emb_df], axis=1))
    return pd.concat(parts, ignore_index=True)


def build_train_args(cfg: dict, feature_csv: Path):
    import argparse as _argparse

    return _argparse.Namespace(
        datasets=["RAW"],
        seeds=cfg["seeds"],
        threshold_pct=cfg["threshold_pct"],
        val_ratio=cfg["val_ratio"],
        epochs=cfg["epochs"],
        patience=cfg["patience"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        gate_aux=cfg["gate_aux"],
        expert_aux=cfg["expert_aux"],
        gstar_aux=cfg["gstar_aux"],
        high_weight=cfg["high_weight"],
        trunk_dim=cfg["trunk_dim"],
        head_dim=cfg["head_dim"],
        dropout=cfg["dropout"],
        adapter_type=cfg["adapter_type"],
        adapter_hidden=cfg["adapter_hidden"],
        adapter_bottleneck=cfg["adapter_bottleneck"],
        adapter_dropout=cfg["adapter_dropout"],
        adapter_ln_mode=cfg["adapter_ln_mode"],
        adapter_alpha_init=cfg["adapter_alpha_init"],
        adapter_alpha_learnable=cfg["adapter_alpha_learnable"],
        use_metadata=cfg.get("use_metadata", False),
        metadata_fields=cfg.get("metadata_fields", ["month", "season"]),
        metadata_hidden=cfg.get("metadata_hidden", 16),
        metadata_mode=cfg.get("metadata_mode", "concat"),
        core_type=cfg["core_type"],
        raw_feature_csv=str(feature_csv),
        ce_feature_csv="",
        ensemble_mode=cfg["ensemble_mode"],
        ensemble_weight_power=cfg["ensemble_weight_power"],
        ensemble_topk=cfg.get("ensemble_topk", 5),
        tag=cfg["tag"],
        explicit_split=True,
    )


def save_split_files(manifest: pd.DataFrame, output_dir: Path) -> dict:
    split_norm = manifest["split"].astype(str).str.lower().str.strip()
    train_pool = manifest[split_norm == "train"].copy()
    val_pool = manifest[split_norm == "val"].copy()
    test_pool = manifest[split_norm == "test"].copy()
    if train_pool.empty:
        raise ValueError("no rows found for split=train")
    if val_pool.empty:
        raise ValueError("no rows found for split=val")
    if test_pool.empty:
        raise ValueError("no rows found for split=test")

    train_csv = output_dir / "train_pool.csv"
    val_csv = output_dir / "val.csv"
    test_csv = output_dir / "test.csv"
    train_pool.to_csv(train_csv, index=False)
    val_pool.to_csv(val_csv, index=False)
    test_pool.to_csv(test_csv, index=False)

    summary = {
        "split_mode": "explicit_train_val_test_folders",
        "n_train_pool": int(len(train_pool)),
        "n_val": int(len(val_pool)),
        "n_test": int(len(test_pool)),
        "val_split_note": "Validation is fixed by the explicit val/ folder.",
        "train_pool_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
    }
    write_json(output_dir / "split_summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the site pipeline end-to-end.")
    ap.add_argument("--site-dir", default="", help="Single site folder containing roi.json, labels, and train/val/test image folders.")
    ap.add_argument("--site-list-csv", default="", help="CSV containing site_id, site_dir, and optional labels_csv.")
    ap.add_argument(
        "--manifest-csv",
        default="",
        help="Optional prebuilt manifest CSV. When provided, skip image/label matching and use this filtered split manifest directly.",
    )
    ap.add_argument("--labels-csv", default="", help="Optional labels table. If omitted, auto-detect inside site-dir.")
    ap.add_argument("--roi-path", default="", help="Optional single ROI override. By default each site uses its own <site_dir>/roi.json.")
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output folder. Default: <site-dir>/pipeline_run_best or <site-list-parent>/pipeline_run_best",
    )
    ap.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Best-pipeline JSON config to use.",
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

    if not args.site_dir and not args.site_list_csv and not args.manifest_csv:
        raise ValueError("please provide --site-dir, --site-list-csv, or --manifest-csv")

    split_dirs = {"train": args.train_dir_name, "val": args.val_dir_name, "test": args.test_dir_name}
    site_root_for_output = None
    if args.manifest_csv:
        manifest_csv_input = Path(args.manifest_csv).resolve()
        manifest = pd.read_csv(manifest_csv_input)
        unmatched = pd.DataFrame()
        site_root_for_output = manifest_csv_input.parent
        labels_csv_for_meta = ""
    elif args.site_list_csv:
        site_list_csv = Path(args.site_list_csv).resolve()
        manifest, unmatched = build_multi_site_dataset(
            site_list_csv=site_list_csv,
            match_tolerance_minutes=args.match_tolerance_minutes,
            split_dirs=split_dirs,
        )
        site_list_df = pd.read_csv(site_list_csv)
        site_root_for_output = site_list_csv.parent
        labels_csv_for_meta = ""
    else:
        site_dir = Path(args.site_dir).resolve()
        labels_csv = Path(args.labels_csv).resolve() if args.labels_csv else auto_detect_labels_csv(site_dir)
        if labels_csv is None:
            raise FileNotFoundError("could not auto-detect a labels CSV inside site-dir")
        manifest, unmatched = build_site_dataset_from_splits(
            site_dir=site_dir,
            labels_csv=labels_csv,
            match_tolerance_minutes=args.match_tolerance_minutes,
            split_dirs=split_dirs,
            site_id=site_dir.name,
        )
        site_root_for_output = site_dir
        labels_csv_for_meta = str(labels_csv)

    roi_path = Path(args.roi_path).resolve() if args.roi_path else None
    if roi_path is not None and not roi_path.exists():
        raise FileNotFoundError(f"roi file not found: {roi_path}")
    if roi_path is None:
        if "roi_path" not in manifest.columns:
            raise ValueError("manifest has no roi_path column")
        missing_roi = [p for p in manifest["roi_path"].dropna().unique().tolist() if not Path(str(p)).exists()]
        if missing_roi:
            raise FileNotFoundError(f"some site roi.json files are missing, first missing path: {missing_roi[0]}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (site_root_for_output / "pipeline_run_best").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_best_config(Path(args.config).resolve())
    manifest_csv = output_dir / "manifest.csv"
    unmatched_csv = output_dir / "unmatched_images.csv"
    manifest.to_csv(manifest_csv, index=False)
    unmatched.to_csv(unmatched_csv, index=False)

    split_summary = save_split_files(manifest=manifest, output_dir=output_dir)

    print("[1/3] Extracting ROI features with best backbone...", flush=True)
    feature_df = build_feature_table(
        meta=manifest,
        roi_path=roi_path,
        backbone=cfg["backbone"],
        batch_size=cfg["extract_batch_size"],
        refine_water_mask=cfg.get("refine_water_mask", True),
    )
    feature_csv = output_dir / "features_convnext_tiny.csv"
    feature_df.to_csv(feature_csv, index=False)

    trainer_mod = load_module("pipeline_trainer_mod", BUNDLE_DIR / "run_v67_on_backbone_features_adapter.py")
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[2/3] Training and evaluating best model... device={device}", flush=True)
    train_args = build_train_args(cfg, feature_csv)
    ds = trainer_mod.load_df(feature_csv, apply_year_filter=not bool(train_args.explicit_split))
    summary, pred_df, fit_logs = trainer_mod.evaluate_dataset("RAW", ds, args=train_args, device=device)

    metrics_csv = output_dir / "metrics.csv"
    predictions_csv = output_dir / "predictions.csv"
    seed_logs_csv = output_dir / "seed_logs.csv"
    pd.DataFrame([summary]).to_csv(metrics_csv, index=False)
    pred_df.to_csv(predictions_csv, index=False)
    pd.DataFrame(fit_logs).to_csv(seed_logs_csv, index=False)

    print("[3/3] Writing run metadata...", flush=True)
    run_config = {
        "site_dir": args.site_dir,
        "site_list_csv": args.site_list_csv,
        "manifest_csv_input": args.manifest_csv,
        "labels_csv": labels_csv_for_meta,
        "roi_path": str(roi_path),
        "output_dir": str(output_dir),
        "manifest_csv": str(manifest_csv),
        "unmatched_csv": str(unmatched_csv),
        "feature_csv": str(feature_csv),
        "metrics_csv": str(metrics_csv),
        "predictions_csv": str(predictions_csv),
        "seed_logs_csv": str(seed_logs_csv),
        "device": str(device),
        "match_tolerance_minutes": args.match_tolerance_minutes,
        "best_config": cfg,
        "split_summary": split_summary,
    }
    write_json(output_dir / "run_config.json", run_config)

    print("\nDone.")
    print(json.dumps(
        {
            "soft_R2": summary["soft_R2"],
            "hard_R2": summary["hard_R2"],
            "n_train_pool": split_summary["n_train_pool"],
            "n_test": split_summary["n_test"],
            "output_dir": str(output_dir),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
