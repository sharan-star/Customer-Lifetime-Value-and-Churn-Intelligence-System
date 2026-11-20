"""
Dataset loader for the Customer Lifetime Value & Churn Intelligence System.

Loads the **real** Kaggle datasets from ``data/raw`` (populated by
``scripts/download_data.py``), falling back to a kagglehub download and finally
to a reproducible synthetic dataset so the demo still runs offline.

* IBM Telco Customer Churn  → churn, survival, CLV, segmentation.
* Hillstrom MineThatData    → real randomized uplift modelling.

All file paths are built with ``os.path.join`` so the code works from any
working directory (notebook, script, or Streamlit app).
"""

import os
import glob
import logging
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project root importable regardless of the current working directory
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import RAW_DATA_DIR  # noqa: E402

logger = logging.getLogger(__name__)

# ── Canonical raw file names (resolved with os.path.join) ────────────────────
TELCO_RAW_FILENAME = "telco_customer_churn.csv"
HILLSTROM_RAW_FILENAME = "hillstrom_email_marketing.csv"


def telco_raw_path() -> str:
    """Absolute path to the local raw Telco CSV."""
    return os.path.join(str(RAW_DATA_DIR), TELCO_RAW_FILENAME)


def hillstrom_raw_path() -> str:
    """Absolute path to the local raw Hillstrom CSV."""
    return os.path.join(str(RAW_DATA_DIR), HILLSTROM_RAW_FILENAME)


# ---------------------------------------------------------------------------
# Synthetic dataset generator (fallback)
# ---------------------------------------------------------------------------

def _generate_synthetic_dataset(n_rows: int = 7043, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic Telco churn dataset matching the IBM schema."""
    rng = np.random.default_rng(seed)

    customer_ids = [f"{i:04d}-{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for i in range(1, n_rows + 1)]

    gender = rng.choice(["Male", "Female"], n_rows)
    senior_citizen = rng.choice([0, 1], n_rows, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n_rows, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n_rows, p=[0.30, 0.70])

    tenure = rng.integers(0, 73, n_rows)

    phone_service = rng.choice(["Yes", "No"], n_rows, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], n_rows, p=[0.42, 0.58]),
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], n_rows, p=[0.34, 0.44, 0.22]
    )

    def _internet_dep(yes_prob: float) -> np.ndarray:
        return np.where(
            internet_service == "No",
            "No internet service",
            rng.choice(["Yes", "No"], n_rows, p=[yes_prob, 1 - yes_prob]),
        )

    online_security = _internet_dep(0.29)
    online_backup = _internet_dep(0.34)
    device_protection = _internet_dep(0.34)
    tech_support = _internet_dep(0.29)
    streaming_tv = _internet_dep(0.38)
    streaming_movies = _internet_dep(0.39)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n_rows, p=[0.55, 0.21, 0.24]
    )
    paperless_billing = rng.choice(["Yes", "No"], n_rows, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n_rows,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # MonthlyCharges depend on services
    base_charge = 20.0
    monthly = base_charge + np.zeros(n_rows)
    monthly += np.where(phone_service == "Yes", rng.uniform(0, 10, n_rows), 0)
    monthly += np.where(multiple_lines == "Yes", rng.uniform(5, 10, n_rows), 0)
    monthly += np.where(internet_service == "DSL", rng.uniform(15, 25, n_rows), 0)
    monthly += np.where(internet_service == "Fiber optic", rng.uniform(30, 50, n_rows), 0)
    monthly += np.where(online_security == "Yes", rng.uniform(5, 10, n_rows), 0)
    monthly += np.where(online_backup == "Yes", rng.uniform(5, 10, n_rows), 0)
    monthly += np.where(device_protection == "Yes", rng.uniform(5, 10, n_rows), 0)
    monthly += np.where(tech_support == "Yes", rng.uniform(5, 10, n_rows), 0)
    monthly += np.where(streaming_tv == "Yes", rng.uniform(5, 15, n_rows), 0)
    monthly += np.where(streaming_movies == "Yes", rng.uniform(5, 15, n_rows), 0)
    monthly = np.round(monthly, 2)

    total_charges = np.round(monthly * np.maximum(tenure, 1) + rng.normal(0, 5, n_rows), 2)
    total_charges = np.maximum(total_charges, monthly)  # floor
    total_charges_str = np.where(tenure == 0, " ", total_charges.astype(str))

    # Churn – driven by contract, tenure, charges, support
    churn_logit = (
        -1.0
        + 1.5 * (contract == "Month-to-month").astype(float)
        - 0.03 * tenure
        + 0.01 * monthly
        + 0.4 * (internet_service == "Fiber optic").astype(float)
        - 0.5 * (online_security == "Yes").astype(float)
        - 0.5 * (tech_support == "Yes").astype(float)
        + 0.3 * (paperless_billing == "Yes").astype(float)
        + 0.3 * (payment_method == "Electronic check").astype(float)
        + rng.normal(0, 0.3, n_rows)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = np.where(rng.random(n_rows) < churn_prob, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly,
            "TotalCharges": total_charges_str,
            "Churn": churn,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Treatment column for uplift modelling
# ---------------------------------------------------------------------------

def _add_treatment_column(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Add a synthetic 'treatment' column with heterogeneous treatment effects.

    * ~50 % of customers are randomly assigned to treatment (retention offer).
    * "Persuadable" treated customers (month-to-month, tenure < 12, no tech
      support) get their churn flipped from Yes→No with ~15 pp extra
      probability, modelling a real retention effect.
    * "Always churners" (long-term high-value) and "never churners" do not
      respond to treatment.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    # Random 50/50 assignment
    df["treatment"] = rng.choice([0, 1], len(df), p=[0.5, 0.5])

    # Identify persuadable segment
    persuadable = (
        (df["Contract"] == "Month-to-month")
        & (df["tenure"] < 12)
        & (df["TechSupport"].isin(["No", "No internet service"]))
    )

    # Among treated persuadable customers who churned, flip ~15 pp to retained
    mask = persuadable & (df["treatment"] == 1) & (df["Churn"] == "Yes")
    flip = rng.random(mask.sum()) < 0.15
    df.loc[mask, "Churn"] = np.where(flip, "No", "Yes")

    logger.info(
        "Treatment column added – treated: %d, control: %d, persuadable treated: %d",
        (df["treatment"] == 1).sum(),
        (df["treatment"] == 0).sum(),
        (persuadable & (df["treatment"] == 1)).sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _download_via_kagglehub(slug: str, dest: str) -> None:
    """Download *slug* with kagglehub and copy its CSV to *dest*."""
    import kagglehub  # type: ignore

    logger.info("Downloading %s via kagglehub ...", slug)
    dataset_path = kagglehub.dataset_download(slug)
    csv_files = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV found for {slug} under {dataset_path}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(csv_files[0], dest)
    logger.info("Saved %s -> %s", slug, dest)


def load_telco_data(add_synthetic_treatment: bool = False) -> pd.DataFrame:
    """Load the IBM Telco Customer Churn dataset (real data preferred).

    Resolution order:
        1. Local file ``data/raw/telco_customer_churn.csv`` (built with os.path.join).
        2. kagglehub download, cached to the same local path.
        3. Reproducible synthetic dataset (offline fallback only).

    The real churn labels are preserved. A synthetic ``treatment`` column is
    only added when ``add_synthetic_treatment=True``; genuine causal uplift
    analysis uses the Hillstrom dataset via :func:`prepare_uplift_data`.
    """
    path = telco_raw_path()
    df = None

    if os.path.exists(path):
        df = pd.read_csv(path)
        logger.info("Loaded local Telco dataset: %d rows from %s", len(df), path)

    if df is None:
        try:
            _download_via_kagglehub("blastchar/telco-customer-churn", path)
            df = pd.read_csv(path)
            logger.info("Loaded Telco dataset from Kaggle: %d rows", len(df))
        except Exception as exc:  # noqa: BLE001
            logger.warning("kagglehub download failed (%s); using synthetic data.", exc)

    if df is None:
        logger.info("Generating synthetic Telco dataset (7,043 rows) ...")
        df = _generate_synthetic_dataset()

    if add_synthetic_treatment:
        df = _add_treatment_column(df)

    return df


def load_hillstrom_data() -> pd.DataFrame:
    """Load the real Hillstrom MineThatData e-mail dataset (randomized A/B test)."""
    path = hillstrom_raw_path()
    if not os.path.exists(path):
        _download_via_kagglehub("sumittarya/hillstrom-email-marketing", path)
    df = pd.read_csv(path)
    logger.info("Loaded Hillstrom dataset: %d rows from %s", len(df), path)
    return df


def prepare_uplift_data(
    df: pd.DataFrame, outcome: str = "visit"
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Prepare ``(X, treatment, y)`` from a Hillstrom-style dataframe.

    Parameters
    ----------
    df : DataFrame
        Raw Hillstrom data (columns include ``segment`` and the outcomes).
    outcome : str
        Outcome column to model: ``visit``, ``conversion`` or ``spend``.

    Returns
    -------
    (X, treatment, y) : DataFrame, Series, Series
        ``treatment`` is 1 for any e-mail campaign and 0 for the ``No E-Mail``
        control group (a genuine randomized assignment).
    """
    df = df.copy()
    treatment = (df["segment"] != "No E-Mail").astype(int)

    X = pd.DataFrame(index=df.index)
    X["recency"] = df["recency"].astype(float)
    X["history"] = df["history"].astype(float)
    X["mens"] = df["mens"].astype(int)
    X["womens"] = df["womens"].astype(int)
    X["newbie"] = df["newbie"].astype(int)
    dummies = pd.get_dummies(df[["zip_code", "channel"]].astype(str), drop_first=True)
    X = pd.concat([X, dummies], axis=1).astype(float)

    y = df[outcome].astype(float)
    return X, treatment, y


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = load_telco_data()
    print(f"Telco shape: {data.shape}")
    print(f"Telco columns: {list(data.columns)}")
    print(f"Churn distribution:\n{data['Churn'].value_counts()}")

    hillstrom = load_hillstrom_data()
    print(f"\nHillstrom shape: {hillstrom.shape}")
    print(f"Segment distribution:\n{hillstrom['segment'].value_counts()}")
    X, treatment, y = prepare_uplift_data(hillstrom)
    print(f"\nUplift X shape: {X.shape}")
    print(f"Treatment distribution:\n{treatment.value_counts()}")
    print(f"Visit rate by treatment:\n{y.groupby(treatment).mean()}")
