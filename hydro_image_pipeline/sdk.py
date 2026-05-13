"""Small SDK helpers for the hydro image discharge pipeline.

The pipeline is still organized around command-line stages. This module gives
Python users a stable place to discover the default package, data, config, and
result paths without hard-coding them in notebooks or downstream scripts.
"""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PIPELINE_ROOT = PACKAGE_ROOT.parent
DATA_ROOT = PIPELINE_ROOT / "data"
RESULTS_ROOT = PIPELINE_ROOT / "results"
CONFIG_ROOT = PACKAGE_ROOT / "configs"

SITE_LIST_CSV = CONFIG_ROOT / "site_list.csv"
BEST_CONFIG_JSON = CONFIG_ROOT / "best_pipeline_config.json"
ROI_VARIANTS_ROOT = DATA_ROOT / "roi_variants"
PIPELINE_SITE_LIST_CSV = DATA_ROOT / "pipeline_site_list.csv"
MOE_ROI_RESULTS_ROOT = RESULTS_ROOT / "moe_roi_variants"
SKLEARN_BASELINE_RESULTS_ROOT = RESULTS_ROOT / "sklearn_baselines"


def paths() -> dict[str, Path]:
    """Return the standard paths used by the packaged pipeline."""

    return {
        "pipeline_root": PIPELINE_ROOT,
        "package_root": PACKAGE_ROOT,
        "data_root": DATA_ROOT,
        "results_root": RESULTS_ROOT,
        "config_root": CONFIG_ROOT,
        "site_list_csv": SITE_LIST_CSV,
        "best_config_json": BEST_CONFIG_JSON,
        "roi_variants_root": ROI_VARIANTS_ROOT,
        "pipeline_site_list_csv": PIPELINE_SITE_LIST_CSV,
        "moe_roi_results_root": MOE_ROI_RESULTS_ROOT,
        "sklearn_baseline_results_root": SKLEARN_BASELINE_RESULTS_ROOT,
    }
