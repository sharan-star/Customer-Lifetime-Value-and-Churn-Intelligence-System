"""Streamlit dashboard for churn, survival, uplift, and CLV decisions."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import APP_ICON, APP_TITLE  # noqa: E402
from src.churn_model.churn_classifier import ChurnClassifier  # noqa: E402
from src.clv_segmentation.clv_calculator import CLVCalculator  # noqa: E402
from src.data_pipeline.feature_engineering import (  # noqa: E402
    engineer_features,
    prepare_model_data,
)
from src.data_pipeline.load_dataset import load_telco_data  # noqa: E402
from src.survival_analysis.survival_model import SurvivalAnalyzer  # noqa: E402
from src.uplift_modeling.uplift_model import UpliftModeler  # noqa: E402


@st.cache_data(show_spinner="Loading customer data...")
def load_data() -> pd.DataFrame:
    """Load and engineer the configured dataset once per Streamlit session.

    A randomized retention-offer flag is simulated for the Telco book of business
    (``add_synthetic_treatment=True``) so the ROI simulator has per-customer uplift.
    """
    return engineer_features(load_telco_data(add_synthetic_treatment=True))


@st.cache_resource(show_spinner="Training churn model...")
def train_churn_model(df: pd.DataFrame) -> tuple[ChurnClassifier, pd.DataFrame, pd.Series]:
    """Train the classifier and return its model-ready holdout data."""
    classifier, metrics, X_test, y_test = ChurnClassifier.train_and_evaluate(df)
    return classifier, X_test, y_test


@st.cache_resource(show_spinner="Fitting survival and uplift models...")
def train_decision_models(
    df: pd.DataFrame,
) -> tuple[SurvivalAnalyzer, UpliftModeler, pd.DataFrame]:
    """Fit survival and uplift models used by the dashboard."""
    survival = SurvivalAnalyzer.run_full_analysis(df)
    uplift, segments = UpliftModeler.run_full_analysis(df)
    return survival, uplift, segments


def _roi_table(
    df: pd.DataFrame,
    uplift: UpliftModeler,
    offer_cost: float,
    discount: float,
) -> pd.DataFrame:
    """Calculate offer economics for customers predicted to benefit."""
    X, _ = prepare_model_data(df)
    churn_prob = st.session_state["churn_prob"]
    scores = uplift.predict_uplift(X)
    eligible = scores > 0
    economics = df.loc[eligible, ["customerID", "MonthlyCharges"]].copy()
    economics["churn_probability"] = churn_prob[eligible]
    economics["uplift"] = scores[eligible]
    result = CLVCalculator().compute_expected_revenue_impact(
        economics,
        economics["uplift"].to_numpy(),
        offer_cost=offer_cost,
    )
    result["customerID"] = economics["customerID"].to_numpy()
    result["discount_pct"] = discount
    return result


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption("Evidence-based retention: predict risk, estimate lifetime, and target profitable offers.")

    df = load_data()
    classifier, X_test, y_test = train_churn_model(df)
    probabilities = classifier.predict(prepare_model_data(df)[0])
    st.session_state["churn_prob"] = probabilities

    metrics = classifier._evaluate(X_test, y_test)
    _, uplift, _ = train_decision_models(df)
    overview, risk, economics = st.tabs(["Overview", "Retention ROI", "Customer data"])

    with overview:
        columns = st.columns(4)
        columns[0].metric("Customers", f"{len(df):,}")
        columns[1].metric("Observed churn", f"{(df['Churn'] == 'Yes').mean():.1%}")
        columns[2].metric("ROC-AUC", f"{metrics['auc_roc']:.3f}")
        columns[3].metric("Avg monthly charge", f"${df['MonthlyCharges'].mean():,.2f}")
        st.subheader("Predicted churn risk")
        chart = pd.DataFrame({"Churn probability": probabilities})
        st.bar_chart(chart["Churn probability"].value_counts(bins=10).sort_index())

    with risk:
        offer_cost = st.number_input("Offer cost per customer ($)", 0.0, 1000.0, 10.0, 1.0)
        discount = st.slider("Discount shown to customer (%)", 0, 50, 15)
        if st.button("Calculate retention ROI", type="primary"):
            table = _roi_table(df, uplift, offer_cost, discount)
            total_revenue = table["revenue_from_retention"].sum()
            total_cost = offer_cost * len(table)
            net = total_revenue - total_cost
            c1, c2, c3 = st.columns(3)
            c1.metric("Customers targeted", f"{len(table):,}")
            c2.metric("Expected retained revenue", f"${total_revenue:,.0f}")
            c3.metric("Net value", f"${net:,.0f}", delta=f"{(net / total_cost):.1%} ROI" if total_cost else "n/a")
            st.dataframe(table.sort_values("net_value", ascending=False).head(100), use_container_width=True)

    with economics:
        st.dataframe(
            df.assign(churn_probability=probabilities)
            .loc[:, ["customerID", "Contract", "tenure", "MonthlyCharges", "churn_probability", "Churn"]]
            .sort_values("churn_probability", ascending=False),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
