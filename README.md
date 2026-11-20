# Customer Lifetime Value and Churn Intelligence System

This project predicts telecom churn, estimates customer survival, identifies
customers who are likely to respond to retention offers, and compares expected
retained margin with offer cost.

## Setup

```powershell
cd "Customer Lifetime Value and Churn Intelligence System"

# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Download the real Kaggle datasets into data/raw (via kagglehub, no credentials needed)
python scripts\download_data.py

# 4. (optional) Persist the engineered data in PostgreSQL
python scripts\setup_database.py

# 5. Launch the dashboard
streamlit run src\streamlit_app\app.py
```

`scripts/download_data.py` pulls the **real IBM Telco Customer Churn** dataset and the
**Hillstrom MineThatData e-mail** experiment into `data/raw/`. The loader prefers these
local files and only falls back to `kagglehub` (then synthetic data) if they are
missing, so the demo still runs offline. Every file path is assembled with
`os.path.join`, so the notebooks, scripts and the app all work from any working
directory without path conflicts.

To persist data in PostgreSQL, configure `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`
and `DB_PASSWORD`, then run the setup script above (it degrades gracefully when no
database is available).

The analytical modules are organized under `src/`: data loading and feature
engineering, LightGBM/SHAP churn modeling, Lifelines survival analysis,
uplift modeling, and CLV segmentation. Generated plots are written to
`docs/diagrams/`.

## Documentation

See [`docs/architecture.md`](docs/architecture.md) for the system architecture
diagram, the data schema, and the module/API reference.

## Jupyter notebooks

The interactive project workflow is available in:

- `notebooks/01_eda_and_feature_engineering.ipynb`
- `notebooks/02_churn_model_and_shap.ipynb`
- `notebooks/03_survival_uplift_clv_and_segments.ipynb`
- `notebooks/04_end_to_end_training.ipynb`

The notebooks call the reusable Python modules; the `.py` files remain in
place because Streamlit, scripts, and notebooks all share the same logic.

## Datasets

- **IBM Telco Customer Churn** (`data/raw/telco_customer_churn.csv`) - the primary
  dataset for churn classification, survival analysis, CLV and segmentation. It
  contains customer tenure, services, contract details, charges and churn labels.
- **Hillstrom MineThatData E-Mail Campaign** (`data/raw/hillstrom_email_marketing.csv`) -
  a **real randomized A/B test** used to validate the uplift modeling methodology
  (CausalML) against genuine causal effects.

The Telco dataset has no randomized retention-offer flag, so the ROI simulator
attaches a *simulated* offer assignment (`add_synthetic_treatment=True`) purely to
demonstrate per-customer uplift economics. Genuine causal uplift is estimated from the
randomized Hillstrom experiment (see notebook 03). Do not claim causal offer
effectiveness from the Telco dataset unless treatment was randomized.

## Moving from prototype to production

1. Replace the synthetic fallback with a versioned PostgreSQL training extract.
2. Freeze the training snapshot and keep train/validation/test splits separate.
3. Store the model, metrics, feature list, data version, and training timestamp
   together under `models/`.
4. Validate calibration, recall at the retention budget, and uplift on a
   randomized campaign holdout.
5. Add automated tests, scheduled retraining, drift monitoring, and approval
   before exposing recommendations to business users.
6. Deploy the Streamlit dashboard only after the data and model artifacts are
   reproducible outside the notebook.
