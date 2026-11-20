"""
Central configuration for the Customer Lifetime Value & Churn Intelligence System.
"""
import os
from pathlib import Path

# ─── Project Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def _load_dotenv() -> None:
    """Load ``KEY=VALUE`` pairs from ``<project_root>/.env`` into the environment.

    Existing environment variables always win, so shell overrides still work.
    """
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# ─── PostgreSQL ─────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "churn_intelligence"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

DATABASE_URL = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

# ─── Model Parameters ──────────────────────────────────────────────────────
LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 300,
    "early_stopping_rounds": 30,
    "random_state": 42,
}

# ─── Feature Engineering ────────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

NUMERICAL_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
]

TARGET_COLUMN = "Churn"

# ─── CLV Parameters ────────────────────────────────────────────────────────
DISCOUNT_RATE_ANNUAL = 0.10  # 10% annual discount rate
AVERAGE_MONTHLY_MARGIN = 0.30  # 30% margin on monthly charges

# ─── Clustering ─────────────────────────────────────────────────────────────
N_CLUSTERS = 4

# ─── Streamlit App ──────────────────────────────────────────────────────────
APP_TITLE = "Customer Lifetime Value & Churn Intelligence System"
APP_ICON = "📊"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
