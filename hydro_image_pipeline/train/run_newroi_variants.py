from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


BUNDLE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = BUNDLE_DIR.parent
PIPELINE_ROOT = PACKAGE_ROOT.parent
DEFAULT_SITE_LIST = PIPELINE_ROOT / "data" / "pipeline_site_list.csv"
DEFAULT_NEWROI_ROOT = PIPELINE_ROOT / "data" / "roi_variants"
DEFAULT_OUTPUT_ROOT = PIPELINE_ROOT / "results" / "moe_roi_variants"
DEFAULT_WORK_ROOT = DEFAULT_OUTPUT_ROOT / "_inputs"
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "best_pipeline_config.json"
VARIANTS = ("p05", "p50", "p95")


@dataclass(frozen=True)
class RoiTask:
    site_id: str
    variant: str
    site_dir: Path
    roi_path: Path
    source_manifest: Path
    variant_manifest: Path
    output_dir: Path


def load_site_rows(site_list_csv: Path) -> list[dict[str, str]]:
    with site_list_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"site list is empty: {site_list_csv}")
    missing = {"site_id", "site_dir"} - set(rows[0])
    if missing:
        raise ValueError(f"{site_list_csv} is missing columns: {sorted(missing)}")
    return rows


def validate_roi_json(path: Path) -> None:
    obj = json.loads(path.read_text(encoding="utf-8"))
    shapes = obj.get("shapes") or []
    if not any(shape.get("points") for shape in shapes):
        raise ValueError(f"ROI file has no polygon points: {path}")
    if "imageWidth" not in obj or "imageHeight" not in obj:
        raise ValueError(f"ROI file is missing imageWidth/imageHeight: {path}")


def build_tasks(
    site_list_csv: Path,
    newroi_root: Path,
    output_root: Path,
    work_root: Path,
    selected_sites: set[str] | None,
    selected_variants: set[str] | None,
) -> list[RoiTask]:
    tasks: list[RoiTask] = []
    for row in load_site_rows(site_list_csv):
        site_id = str(row["site_id"]).strip()
        if selected_sites and site_id not in selected_sites:
            continue
        site_dir = Path(str(row["site_dir"])).resolve()
        source_manifest = site_dir / "split_manifest_filtered.csv"
        if not source_manifest.exists():
            raise FileNotFoundError(f"missing filtered split manifest for {site_id}: {source_manifest}")
        for variant in VARIANTS:
            if selected_variants and variant not in selected_variants:
                continue
            roi_path = (newroi_root / site_id / f"{site_id}_{variant}.json").resolve()
            if not roi_path.exists():
                raise FileNotFoundError(f"missing {variant} ROI for {site_id}: {roi_path}")
            validate_roi_json(roi_path)
            variant_manifest = (work_root / variant / f"{site_id}_manifest.csv").resolve()
            output_dir = (output_root / variant / site_id).resolve()
            tasks.append(
                RoiTask(
                    site_id=site_id,
                    variant=variant,
                    site_dir=site_dir,
                    roi_path=roi_path,
                    source_manifest=source_manifest.resolve(),
                    variant_manifest=variant_manifest,
                    output_dir=output_dir,
                )
            )
    if not tasks:
        raise ValueError("no ROI tasks selected")
    return tasks


def write_variant_manifest(task: RoiTask) -> dict[str, object]:
    df = pd.read_csv(task.source_manifest)
    required = {"image_path", "split", "discharge", "year", "month", "image_time"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{task.source_manifest} is missing columns: {missing}")
    df["roi_path"] = str(task.roi_path)
    df["site_id"] = task.site_id
    df["site_dir"] = str(task.site_dir)
    split_counts = df["split"].astype(str).str.lower().str.strip().value_counts().to_dict()
    for split in ("train", "val", "test"):
        if int(split_counts.get(split, 0)) == 0:
            raise ValueError(f"{task.source_manifest} has no rows for split={split}")
    task.variant_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(task.variant_manifest, index=False)
    return {
        "rows": int(len(df)),
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
    }


def run_task(task: RoiTask, config: Path, dry_run: bool) -> None:
    manifest_summary = write_variant_manifest(task)
    cmd = [
        sys.executable,
        str(BUNDLE_DIR / "run_best_pipeline.py"),
        "--manifest-csv",
        str(task.variant_manifest),
        "--roi-path",
        str(task.roi_path),
        "--output-dir",
        str(task.output_dir),
        "--config",
        str(config),
    ]
    print(
        json.dumps(
            {
                "site_id": task.site_id,
                "variant": task.variant,
                "roi_path": str(task.roi_path),
                "source_manifest": str(task.source_manifest),
                "variant_manifest": str(task.variant_manifest),
                "output_dir": str(task.output_dir),
                **manifest_summary,
                "cmd": cmd,
            },
            indent=2,
        ),
        flush=True,
    )
    if dry_run:
        return
    task.output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run p05/p50/p95 newROI variants for every site using filtered split manifests."
    )
    parser.add_argument("--site-list-csv", type=Path, default=DEFAULT_SITE_LIST)
    parser.add_argument("--newroi-root", type=Path, default=DEFAULT_NEWROI_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--site-id", action="append", default=[], help="Restrict to one site; can repeat.")
    parser.add_argument("--variant", choices=VARIANTS, action="append", default=[], help="Restrict ROI variant; can repeat.")
    parser.add_argument("--task-index", type=int, default=None, help="Run one zero-based task from the selected task list.")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_sites = set(args.site_id) if args.site_id else None
    selected_variants = set(args.variant) if args.variant else None
    tasks = build_tasks(
        site_list_csv=args.site_list_csv.resolve(),
        newroi_root=args.newroi_root.resolve(),
        output_root=args.output_root.resolve(),
        work_root=args.work_root.resolve(),
        selected_sites=selected_sites,
        selected_variants=selected_variants,
    )

    if args.list_tasks:
        for idx, task in enumerate(tasks):
            print(f"{idx}\t{task.variant}\t{task.site_id}\t{task.roi_path}\t{task.output_dir}")
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

    for task in tasks:
        run_task(task=task, config=args.config.resolve(), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
