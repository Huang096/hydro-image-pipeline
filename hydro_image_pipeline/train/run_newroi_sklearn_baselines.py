from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_ROOT = PIPELINE_ROOT / "results" / "moe_roi_variants"
DEFAULT_OUTPUT_ROOT = PIPELINE_ROOT / "results" / "sklearn_baselines"
VARIANTS = ("p05", "p50", "p95")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def load_features(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError(f"no embedding columns found in {path}")
    required = {"split", "discharge", "image_path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    df["discharge"] = pd.to_numeric(df["discharge"], errors="coerce")
    df = df[df["discharge"].notna()].copy()
    return df, emb_cols


def fit_predict(model_name: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray):
    if model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
    elif model_name == "svm":
        model = make_pipeline(
            StandardScaler(),
            LinearSVR(C=1.0, epsilon=0.05, max_iter=20000, random_state=42),
        )
    else:
        raise ValueError(f"unknown model: {model_name}")
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return model, pred


def run_one(feature_csv: Path, output_dir: Path, site_id: str, variant: str) -> list[dict[str, object]]:
    df, emb_cols = load_features(feature_csv)
    train_df = df[df["split"].isin(["train", "val"])].copy()
    test_df = df[df["split"] == "test"].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(f"missing train/val or test rows in {feature_csv}")

    x_train = train_df[emb_cols].to_numpy(dtype=np.float32)
    y_train = train_df["discharge"].to_numpy(dtype=np.float64)
    x_test = test_df[emb_cols].to_numpy(dtype=np.float32)
    y_test = test_df["discharge"].to_numpy(dtype=np.float64)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name in ("random_forest", "svm"):
        _, pred = fit_predict(model_name, x_train, y_train, x_test)
        metric = metrics_dict(y_test, pred)
        rows.append(
            {
                "site_id": site_id,
                "variant": variant,
                "model": model_name,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "n_features": int(len(emb_cols)),
                **metric,
            }
        )
        pred_df = test_df[
            [c for c in ["image_path", "image_time", "year", "month", "discharge"] if c in test_df.columns]
        ].copy()
        pred_df = pred_df.rename(columns={"discharge": "discharge_true"})
        pred_df["pred"] = pred
        pred_df["model"] = model_name
        pred_df.to_csv(output_dir / f"predictions_{model_name}.csv", index=False)

    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "site_id": site_id,
                "variant": variant,
                "feature_csv": str(feature_csv),
                "output_dir": str(output_dir),
                "models": ["random_forest", "svm"],
                "train_splits": ["train", "val"],
                "test_split": "test",
                "target": "discharge",
                "n_embedding_features": len(emb_cols),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows


def build_tasks(feature_root: Path, output_root: Path) -> list[tuple[str, str, Path, Path]]:
    tasks = []
    for variant in VARIANTS:
        variant_dir = feature_root / variant
        if not variant_dir.exists():
            continue
        for site_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
            site = site_dir.name
            feature_csv = site_dir / "features_convnext_tiny.csv"
            if not feature_csv.exists():
                continue
            output_dir = output_root / variant / site
            tasks.append((site, variant, feature_csv, output_dir))
    if not tasks:
        raise FileNotFoundError(f"no features_convnext_tiny.csv files found under {feature_root}")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RandomForest and SVM baselines on newROI ConvNeXt embeddings.")
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()

    tasks = build_tasks(args.feature_root.resolve(), args.output_root.resolve())
    if args.list_tasks:
        for idx, (site, variant, feature_csv, output_dir) in enumerate(tasks):
            print(f"{idx}\t{variant}\t{site}\t{feature_csv}\t{output_dir}")
        return
    if args.task_index is not None:
        if args.task_index < 0 or args.task_index >= len(tasks):
            print(
                json.dumps(
                    {
                        "status": "skipped",
                        "reason": "task index out of range",
                        "task_index": args.task_index,
                        "n_tasks": len(tasks),
                    },
                    indent=2,
                )
            )
            return
        tasks = [tasks[args.task_index]]

    all_rows = []
    for site, variant, feature_csv, output_dir in tasks:
        print(f"[baseline] site={site} variant={variant} features={feature_csv}", flush=True)
        all_rows.extend(run_one(feature_csv, output_dir, site, variant))

    args.output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(args.output_root / "metrics_summary.csv", index=False)
    print(json.dumps({"n_tasks": len(tasks), "output_root": str(args.output_root.resolve())}, indent=2))


if __name__ == "__main__":
    main()
