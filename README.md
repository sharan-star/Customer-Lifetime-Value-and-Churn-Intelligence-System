# Customer Lifetime Value & Churn Intelligence System

An end-to-end analytics workspace for telecom retention decisions. The project combines churn prediction, customer survival analysis, causal uplift modeling, customer lifetime value (CLV), and segmentation to help answer:

- **Who is most likely to churn?**
- **How long is a customer expected to remain active?**
- **Which customers are likely to respond to a retention offer?**
- **Is the expected retained value greater than the offer cost?**

The project includes reusable Python modules, Jupyter notebooks, generated analytical artifacts, and an interactive Streamlit dashboard.

## What the system does

1. Loads the IBM Telco Customer Churn dataset and engineers domain features.
2. Trains a LightGBM classifier to estimate churn probability.
3. Explains churn predictions with SHAP.
4. Estimates customer survival with Kaplan-Meier and Cox proportional-hazards models.
5. Estimates heterogeneous offer effects with uplift modeling.
6. Calculates discounted CLV using monthly revenue and expected tenure.
7. Groups customers into behavioral segments with KMeans.
8. Simulates retention-offer economics and ranks customers by expected net value.

## Quick start

### Requirements

- Python 3.10+ recommended
- Optional: PostgreSQL for persistence
- Internet access is only required when the raw datasets are not already present and KaggleHub needs to download them.

### Install

```powershell
cd "Customer Lifetime Value and Churn Intelligence System"

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Download the datasets

```powershell
python scripts\download_data.py
```

This downloads two public datasets into `data/raw/`:

- **IBM Telco Customer Churn** — churn, survival, CLV, and segmentation.
- **Hillstrom MineThatData E-Mail Campaign** — randomized campaign data for uplift analysis.

The application and notebooks prefer local CSV files. If the Telco file is unavailable, the loader attempts KaggleHub and then creates a reproducible synthetic fallback so the workflow can still run offline.

### Prepare data and optionally load PostgreSQL

```powershell
python scripts\setup_database.py
```

The script loads and engineers the Telco data, writes processed CSV output, and attempts to load the result into PostgreSQL. Database errors are logged and do not prevent the local CSV workflow from completing.

Configure PostgreSQL with environment variables or a project-root `.env` file:

```text
DB_HOST=localhost
DB_PORT=5432
DB_NAME=churn_intelligence
DB_USER=postgres
DB_PASSWORD=postgres
```

### Launch the dashboard

```powershell
streamlit run src\streamlit_app\app.py
```

The dashboard provides:

- Overview metrics and predicted churn-risk distribution.
- A **Retention ROI** simulator with configurable offer cost and discount.
- A customer table sorted by predicted churn probability.
- Expected retained revenue, net value, and ROI for customers with positive predicted uplift.

## Project workflow

```text
data/raw/
    ↓
load_dataset + feature_engineering
    ↓
EDA ── churn model ── SHAP explanations
  ├── survival analysis
  ├── uplift modeling
  └── CLV + customer segmentation
    ↓
data/processed/ + models/ + docs/diagrams/
    ↓
Streamlit dashboard / notebooks
```

### Main components

| Component | Location | Purpose |
|---|---|---|
| Data loading and features | `src/data_pipeline/` | Load datasets and create model-ready features |
| Exploratory analysis | `src/eda/` | Summaries and diagnostic visualizations |
| Churn model | `src/churn_model/` | LightGBM classification and SHAP explanations |
| Survival analysis | `src/survival_analysis/` | Kaplan-Meier and Cox PH modeling |
| Uplift modeling | `src/uplift_modeling/` | Treatment-effect estimation and response segments |
| CLV and segmentation | `src/clv_segmentation/` | Discounted CLV, offer economics, and KMeans profiles |
| Dashboard | `src/streamlit_app/app.py` | Interactive decision interface |
| Configuration | `config/settings.py` | Paths, model parameters, database settings, and CLV assumptions |

## Notebooks

Run the notebooks in order for the full analytical workflow:

1. `notebooks/01_eda_and_feature_engineering.ipynb`
2. `notebooks/02_churn_model_and_shap.ipynb`
3. `notebooks/03_survival_uplift_clv_and_segments.ipynb`
4. `notebooks/04_end_to_end_training.ipynb`

The notebooks use the same reusable modules as the dashboard and scripts.

## Outputs

### Processed data

- `data/processed/telco_churn_processed.csv` — engineered Telco records.
- `data/processed/telco_churn_scored.csv` — records with model scores and segments.
- `data/processed/training_snapshot.csv` — training data snapshot, including treatment where applicable.
- `data/processed/retention_decisions.csv` — customer-level retention economics and recommendations.

### Model artifacts

- `models/churn_lgbm_model.pkl` — serialized LightGBM model bundle.
- `models/churn_metrics.json` — saved classifier metrics.

### Visualizations

Generated charts are stored in `docs/diagrams/`, including churn distributions, contract analysis, survival curves, SHAP explanations, uplift segments, and Qini curves.

See [`docs/architecture.md`](docs/architecture.md) for the architecture diagram, data schema, feature inventory, and Python API reference.

## Data and causal-modeling notes

- The Telco dataset contains observed churn labels but **does not contain randomized retention-offer assignments**.
- The dashboard adds a reproducible synthetic treatment assignment only to demonstrate per-customer uplift and ROI calculations.
- Causal offer effectiveness should not be claimed from the Telco data. The Hillstrom email dataset provides the randomized treatment/control experiment used to validate the uplift workflow.
- Positive uplift means the model estimates that treatment reduces churn probability. Customers with positive uplift are eligible for the dashboard's ROI simulation.
- CLV uses a discounted cash-flow approximation with a 10% annual discount rate and a 30% monthly margin assumption by default. These assumptions are configurable in `config/settings.py`.

## Recommended operating practices

For reliable business use:

1. Replace the synthetic fallback and treatment simulation with versioned production data and randomized campaign outcomes.
2. Freeze reproducible training snapshots and maintain separate train, validation, and test data.
3. Version models together with feature definitions, metrics, data versions, and training timestamps.
4. Validate calibration, budget-constrained recall, uplift, and ROI on a randomized campaign holdout.
5. Add automated tests, scheduled retraining, drift monitoring, access controls, and approval workflows.
6. Deploy the dashboard only after the pipeline and model artifacts are reproducible outside notebooks.

## Conclusion

This project brings together predictive modeling, survival analysis, uplift modeling, CLV estimation, and customer segmentation in one practical retention analytics workflow. It provides the foundation for turning customer risk and value signals into measurable, data-informed retention decisions.

Review the original IBM Telco and Hillstrom dataset terms before redistribution or commercial use.
