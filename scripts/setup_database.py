#!/usr/bin/env python
"""
Setup script for the Customer Lifetime Value & Churn Intelligence System.

1. Loads the Telco Customer Churn dataset.
2. Runs feature engineering.
3. Saves raw + processed CSVs.
4. Attempts to load into PostgreSQL (graceful fallback).
5. Prints summary statistics.
"""

import logging
import sys
from pathlib import Path

# ── Make the project root importable ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR  # noqa: E402
from src.data_pipeline.load_dataset import load_telco_data  # noqa: E402
from src.data_pipeline.feature_engineering import (  # noqa: E402
    engineer_features,
    prepare_model_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    # ── 1. Load dataset ─────────────────────────────────────────────────
    logger.info("Loading Telco Customer Churn dataset …")
    df_raw = load_telco_data()
    logger.info("Raw dataset shape: %s", df_raw.shape)

    # ── 2. Save raw data ────────────────────────────────────────────────
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DATA_DIR / "telco_churn_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    logger.info("Raw data saved → %s", raw_path)

    # ── 3. Feature engineering ──────────────────────────────────────────
    logger.info("Running feature engineering …")
    df_processed = engineer_features(df_raw)

    # ── 4. Save processed data ──────────────────────────────────────────
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DATA_DIR / "telco_churn_processed.csv"
    df_processed.to_csv(processed_path, index=False)
    logger.info("Processed data saved → %s", processed_path)

    # ── 5. PostgreSQL load (graceful fallback) ──────────────────────────
    try:
        from src.data_pipeline.db_manager import DatabaseManager  # noqa: E402

        db = DatabaseManager()
        db.create_database()
        db.load_data_to_db(df_processed)

        # Quick verification
        count_df = db.execute_query("SELECT COUNT(*) AS cnt FROM customers")
        logger.info("PostgreSQL verification – rows in 'customers': %s", count_df.iloc[0, 0])
        db.close()
    except Exception as exc:
        logger.warning("PostgreSQL load skipped: %s", exc)

    # ── 6. Summary statistics ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Rows            : {len(df_processed)}")
    print(f"  Columns         : {len(df_processed.columns)}")
    print(f"  Features (raw)  : {len(df_raw.columns)}")
    print(f"  Features (eng.) : {len(df_processed.columns)}")
    print()

    churn_counts = df_raw["Churn"].value_counts()
    print("  Churn distribution:")
    for val, cnt in churn_counts.items():
        pct = cnt / len(df_raw) * 100
        print(f"    {val:>5s}: {cnt:>5d}  ({pct:.1f}%)")

    if "treatment" in df_raw.columns:
        treat_counts = df_raw["treatment"].value_counts()
        print("\n  Treatment distribution:")
        for val, cnt in treat_counts.items():
            pct = cnt / len(df_raw) * 100
            print(f"    {val}: {cnt:>5d}  ({pct:.1f}%)")

    print(f"\n  Avg tenure      : {df_raw['tenure'].mean():.1f} months")
    print(f"  Avg MonthlyChg  : ${df_raw['MonthlyCharges'].mean():.2f}")

    # Model data check
    X, y = prepare_model_data(df_processed)
    print(f"\n  Model matrix X  : {X.shape}")
    print(f"  Target y dist.  : { {0: int((y==0).sum()), 1: int((y==1).sum())} }")
    print("=" * 60)
    print("  Setup complete ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
