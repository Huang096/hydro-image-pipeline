from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


VARIANTS = {
    "p05": 0.05,
    "p50": 0.50,
    "p95": 0.95,
}
PIPELINE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PIPELINE_ROOT / "data"


def safe_value(value: object) -> str:
    text = str(value)
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in "-_.":
            cleaned.append(ch)
        elif ch in ":/ ":
            cleaned.append("-")
    return "".join(cleaned).strip("-") or "unknown"


def find_split_manifest(site_dir: Path) -> Path:
    for name in ("split_manifest.csv", "split_manifest_filtered.csv"):
        path = site_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"no split manifest found in {site_dir}")


def choose_references(site_dir: Path, output_root: Path, overwrite: bool) -> list[dict[str, object]]:
    manifest_path = find_split_manifest(site_dir)
    df = pd.read_csv(manifest_path)
    required = {"image_path", "split", "discharge", "image_time"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{manifest_path} missing columns: {missing}")

    df["split"] = df["split"].astype(str).str.lower().str.strip()
    df["discharge"] = pd.to_numeric(df["discharge"], errors="coerce")
    train = df[(df["split"] == "train") & df["discharge"].notna()].copy()
    if train.empty:
        raise ValueError(f"{manifest_path} has no train rows with valid discharge")

    site_id = site_dir.name
    roi_dir = output_root / site_id
    ref_dir = roi_dir / "reference_images"
    ref_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for variant, q in VARIANTS.items():
        target = float(train["discharge"].quantile(q))
        idx = (train["discharge"] - target).abs().idxmin()
        selected = train.loc[idx].copy()
        src = Path(str(selected["image_path"])).resolve()
        if not src.exists():
            raise FileNotFoundError(f"selected image does not exist: {src}")
        actual = float(selected["discharge"])
        image_time = str(selected.get("image_time", ""))
        suffix = src.suffix.lower() or ".jpg"
        dst_name = (
            f"{site_id}_{variant}"
            f"__target_{safe_value(round(target, 3))}"
            f"__actual_{safe_value(round(actual, 3))}"
            f"__{safe_value(image_time)}{suffix}"
        )
        dst = ref_dir / dst_name
        if overwrite or not dst.exists():
            shutil.copy2(src, dst)
        rows.append(
            {
                "site_id": site_id,
                "variant": variant,
                "quantile": q,
                "target_discharge": target,
                "actual_discharge": actual,
                "image_time": image_time,
                "source_image_path": str(src),
                "reference_image_path": str(dst.resolve()),
                "expected_roi_json": str((roi_dir / f"{site_id}_{variant}.json").resolve()),
                "manifest_path": str(manifest_path.resolve()),
            }
        )

    pd.DataFrame(rows).to_csv(roi_dir / "roi_reference_manifest.csv", index=False)
    (roi_dir / "roi_reference_summary.json").write_text(
        json.dumps(
            {
                "site_id": site_id,
                "site_dir": str(site_dir.resolve()),
                "manifest_path": str(manifest_path.resolve()),
                "reference_dir": str(ref_dir.resolve()),
                "roi_dir": str(roi_dir.resolve()),
                "n_train_rows": int(len(train)),
                "variants": list(VARIANTS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows


def discover_site_dirs(data_root: Path) -> list[Path]:
    sites: list[Path] = []
    for river_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for site_dir in sorted(p for p in river_dir.iterdir() if p.is_dir()):
            if (site_dir / "split_manifest.csv").exists() or (site_dir / "split_manifest_filtered.csv").exists():
                sites.append(site_dir)
    return sites


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select p05/p50/p95 clean training images as ROI drawing references."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATA_ROOT / "roi_variants")
    parser.add_argument("--site-dir", action="append", default=[], help="Optional site directory; can repeat.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.site_dir:
        site_dirs = [Path(p).resolve() for p in args.site_dir]
    else:
        site_dirs = discover_site_dirs(args.data_root.resolve())
    if not site_dirs:
        raise SystemExit(f"no filtered site directories found under {args.data_root}")

    all_rows: list[dict[str, object]] = []
    for idx, site_dir in enumerate(site_dirs, start=1):
        print(f"[{idx}/{len(site_dirs)}] select ROI references for {site_dir.name}", flush=True)
        all_rows.extend(
            choose_references(
                site_dir=site_dir,
                output_root=args.output_root.resolve(),
                overwrite=args.overwrite,
            )
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_root / "roi_reference_manifest.csv"
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(json.dumps({"n_sites": len(site_dirs), "n_references": len(all_rows), "manifest": str(out_csv)}, indent=2))


if __name__ == "__main__":
    main()
