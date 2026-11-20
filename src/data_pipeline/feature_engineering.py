"""
Feature engineering for the Customer Lifetime Value & Churn Intelligence System.

Provides two public functions:
* ``engineer_features``  – adds domain-driven columns to the raw DataFrame.
* ``prepare_model_data`` – encodes categoricals and returns (X, y) ready for
  model training.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or lightly cleaned Telco dataset.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with new columns appended.
    """
    df = df.copy()

    # Ensure TotalCharges is numeric (spaces → NaN → 0 for engineering)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # tenure_group
    bins = [-1, 12, 24, 36, 48, 60, 72]
    labels = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    # avg_monthly_charge
    df["avg_monthly_charge"] = df["TotalCharges"] / df["tenure"].clip(lower=1)

    # charge_increase
    df["charge_increase"] = df["MonthlyCharges"] - df["avg_monthly_charge"]

    # num_services – count of "Yes" across service columns
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    # For InternetService, "DSL" / "Fiber optic" counts as a service
    def _count_services(row: pd.Series) -> int:
        count = 0
        for col in service_cols:
            val = str(row.get(col, "")).strip()
            if col == "InternetService":
                if val not in ("No", ""):
                    count += 1
            else:
                if val == "Yes":
                    count += 1
        return count

    df["num_services"] = df.apply(_count_services, axis=1)

    # has_internet
    df["has_internet"] = (df["InternetService"] != "No").astype(int)

    # has_phone
    df["has_phone"] = (df["PhoneService"] == "Yes").astype(int)

    # contract_risk
    df["contract_risk"] = (df["Contract"] == "Month-to-month").astype(int)

    # auto_pay
    df["auto_pay"] = df["PaymentMethod"].str.contains(
        "automatic", case=False, na=False
    ).astype(int)

    # is_senior_no_support
    df["is_senior_no_support"] = (
        (df["SeniorCitizen"] == 1)
        & (df["TechSupport"].isin(["No", "No internet service"]))
        & (df["OnlineSecurity"].isin(["No", "No internet service"]))
    ).astype(int)

    # high_value
    df["high_value"] = (df["MonthlyCharges"] > 70).astype(int)

    # tenure_x_monthly
    df["tenure_x_monthly"] = df["tenure"] * df["MonthlyCharges"]

    logger.info("Engineered features added – new shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Model-ready data preparation
# ---------------------------------------------------------------------------

def prepare_model_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Encode features and return (X, y) for modelling.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame **with** engineered features (output of ``engineer_features``).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with label-encoded categoricals.
    y : pd.Series
        Binary target (0 = retained, 1 = churned).
    """
    df = df.copy()

    # ── Clean up service columns ────────────────────────────────────────
    replace_map = {"No internet service": "No", "No phone service": "No"}
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace(replace_map)

    # ── TotalCharges → numeric ──────────────────────────────────────────
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_tc = df["TotalCharges"].median()
    df["TotalCharges"].fillna(median_tc, inplace=True)

    # ── Target ──────────────────────────────────────────────────────────
    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    y = df["Churn"]

    # ── Drop non-feature columns ────────────────────────────────────────
    # Treatment is an intervention assignment, not a customer attribute.
    # Excluding it prevents churn and survival models from learning from
    # randomized offer assignment (uplift modelling extracts it separately).
    drop_cols = ["customerID", "Churn", "treatment"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── Label-encode remaining categoricals ─────────────────────────────
    label_encoders = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    logger.info(
        "Model data prepared – X shape: %s, y distribution: %s",
        X.shape,
        y.value_counts().to_dict(),
    )
    return X, y
