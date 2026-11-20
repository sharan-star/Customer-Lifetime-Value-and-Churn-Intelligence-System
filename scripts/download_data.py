#!/usr/bin/env python
"""
Download the real Kaggle datasets used by the project into ``data/raw``.

Datasets
--------
1. IBM Telco Customer Churn          (Kaggle: blastchar/telco-customer-churn)
   → used for churn classification, survival analysis, CLV and segmentation.

2. Hillstrom MineThatData E-Mail     (Kaggle: sumittarya/hillstrom-email-marketing)
   → real *randomized* treatment/outcome data used for uplift modeling.

Downloads are cached by ``kagglehub`` and copied into the project's ``data/raw``
folder so the rest of the pipeline reads a stable, local path built with
``os.path.join``.  No Kaggle credentials are required for these public datasets.
"""

import glob
import os
import shutil
import sys

# ── Project paths (built with os.path.join to avoid path conflicts) ──────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# ── Dataset registry: kaggle slug → destination filename ────────────────────
DATASETS = {
    "telco": {
        "slug": "blastchar/telco-customer-churn",
        "dest": os.path.join(RAW_DATA_DIR, "telco_customer_churn.csv"),
    },
    "hillstrom": {
        "slug": "sumittarya/hillstrom-email-marketing",
        "dest": os.path.join(RAW_DATA_DIR, "hillstrom_email_marketing.csv"),
    },
}


def _find_csv(dataset_path: str) -> str:
    """Return the first CSV found under *dataset_path* (recursive)."""
    csv_files = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found under {dataset_path}")
    return csv_files[0]


def download_dataset(name: str, slug: str, dest: str) -> str:
    """Download *slug* via kagglehub and copy its CSV to *dest*."""
    import kagglehub  # imported lazily so the rest of the project works without it

    print(f"[{name}] downloading {slug} ...")
    dataset_path = kagglehub.dataset_download(slug)
    source_csv = _find_csv(dataset_path)

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.abspath(source_csv) != os.path.abspath(dest):
        shutil.copyfile(source_csv, dest)
    print(f"[{name}] saved -> {dest}")
    return dest


def main() -> int:
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    failures = []
    for name, meta in DATASETS.items():
        try:
            download_dataset(name, meta["slug"], meta["dest"])
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print(f"[{name}] FAILED: {exc}")

    print("\n" + "=" * 60)
    if failures:
        print(f"Completed with failures: {failures}")
        return 1
    print("All datasets downloaded into:", RAW_DATA_DIR)
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
