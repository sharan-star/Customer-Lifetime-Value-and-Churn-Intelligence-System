"""
Survival Analysis Module for Customer Churn.

Uses lifelines library to perform Kaplan-Meier estimation,
Cox Proportional Hazards modeling, and survival predictions.
"""
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter
from lifelines.statistics import logrank_test

from config.settings import PROJECT_ROOT


# Diagram output directory
DIAGRAM_DIR = PROJECT_ROOT / "docs" / "diagrams"
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)


class SurvivalAnalyzer:
    """Survival analysis for customer churn using Kaplan-Meier and Cox PH models."""

    def __init__(self) -> None:
        self.kmf: KaplanMeierFitter | None = None
        self.kmf_groups: dict[str, KaplanMeierFitter] | None = None
        self.cph: CoxPHFitter | None = None
        self.aft: WeibullAFTFitter | None = None

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _prepare_churn_column(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure Churn is numeric 0/1."""
        df = df.copy()
        if df["Churn"].dtype == object:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
        return df

    @staticmethod
    def _encode_for_cox(df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode Contract and InternetService for Cox model."""
        df = df.copy()
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        internet_map = {"No": 0, "DSL": 1, "Fiber optic": 2}

        if "Contract" in df.columns:
            df["Contract_encoded"] = df["Contract"].map(contract_map).fillna(0).astype(int)
        if "InternetService" in df.columns:
            df["InternetService_encoded"] = (
                df["InternetService"].map(internet_map).fillna(0).astype(int)
            )
        return df

    # ── Kaplan-Meier ─────────────────────────────────────────────────────

    def fit_kaplan_meier(
        self,
        durations: pd.Series | np.ndarray,
        event_observed: pd.Series | np.ndarray,
        label: str = "Overall",
    ) -> "SurvivalAnalyzer":
        """Fit a Kaplan-Meier survival estimator.

        Args:
            durations: Tenure (time to event or censoring).
            event_observed: 1 = churned, 0 = censored (still active).
            label: Legend label.

        Returns:
            self for chaining.
        """
        self.kmf = KaplanMeierFitter()
        self.kmf.fit(durations, event_observed=event_observed, label=label)
        return self

    def fit_kaplan_meier_by_group(
        self,
        df: pd.DataFrame,
        duration_col: str = "tenure",
        event_col: str = "Churn",
        group_col: str = "Contract",
    ) -> dict[str, KaplanMeierFitter]:
        """Fit separate Kaplan-Meier estimators per group level.

        Args:
            df: DataFrame with duration, event, and group columns.
            duration_col: Name of the duration column.
            event_col: Name of the event column.
            group_col: Name of the grouping column.

        Returns:
            Dict mapping group value → fitted KaplanMeierFitter.
        """
        df = self._prepare_churn_column(df)
        self.kmf_groups = {}
        for group_val, grp in df.groupby(group_col):
            kmf = KaplanMeierFitter()
            kmf.fit(grp[duration_col], event_observed=grp[event_col], label=str(group_val))
            self.kmf_groups[str(group_val)] = kmf
        return self.kmf_groups

    # ── Cox Proportional Hazards ─────────────────────────────────────────

    _DEFAULT_COVARIATES = [
        "SeniorCitizen",
        "MonthlyCharges",
        "Contract_encoded",
        "InternetService_encoded",
        "num_services",
        "auto_pay",
        "contract_risk",
    ]

    def fit_cox_model(
        self,
        df: pd.DataFrame,
        duration_col: str = "tenure",
        event_col: str = "Churn",
        covariates: list[str] | None = None,
    ) -> "SurvivalAnalyzer":
        """Fit a Cox Proportional Hazards model.

        Args:
            df: Data with duration, event, and covariate columns.
            duration_col: Duration column name.
            event_col: Event column name.
            covariates: Covariate column names (defaults to a sensible set).

        Returns:
            self for chaining.
        """
        df = self._prepare_churn_column(df)
        df = self._encode_for_cox(df)

        if covariates is None:
            covariates = self._DEFAULT_COVARIATES

        # Keep only existing covariates
        available = [c for c in covariates if c in df.columns]
        cols = [duration_col, event_col] + available

        cox_df = df[cols].copy()
        # Drop rows with NaN in covariates
        cox_df = cox_df.dropna()
        # Ensure all numeric
        for c in available:
            cox_df[c] = pd.to_numeric(cox_df[c], errors="coerce")
        cox_df = cox_df.dropna()

        # Ensure tenure >= 1 to avoid zero-duration issues
        cox_df[duration_col] = cox_df[duration_col].clip(lower=1)

        self.cph = CoxPHFitter(penalizer=0.01)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cph.fit(
                cox_df,
                duration_col=duration_col,
                event_col=event_col,
                show_progress=False,
            )
        return self

    # ── Predictions ──────────────────────────────────────────────────────

    def predict_survival(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict survival function for each customer using fitted Cox model.

        Args:
            df: DataFrame with the same covariates used for fitting.

        Returns:
            DataFrame with survival probabilities at each time point.
            Rows = time points, Columns = customers (by index).
        """
        if self.cph is None:
            raise RuntimeError("Cox model not fitted. Call fit_cox_model first.")

        df = self._prepare_churn_column(df)
        df = self._encode_for_cox(df)

        needed = [c for c in self.cph.params_.index if c in df.columns]
        pred_df = df[needed].copy()
        for c in needed:
            pred_df[c] = pd.to_numeric(pred_df[c], errors="coerce")
        pred_df = pred_df.fillna(0)

        return self.cph.predict_survival_function(pred_df)

    def predict_median_survival(self, df: pd.DataFrame) -> pd.Series:
        """Predict median expected remaining tenure for each customer.

        Args:
            df: DataFrame with covariate columns.

        Returns:
            Series of median survival times indexed by customer.
        """
        if self.cph is None:
            raise RuntimeError("Cox model not fitted. Call fit_cox_model first.")

        df = self._prepare_churn_column(df)
        df = self._encode_for_cox(df)

        needed = [c for c in self.cph.params_.index if c in df.columns]
        pred_df = df[needed].copy()
        for c in needed:
            pred_df[c] = pd.to_numeric(pred_df[c], errors="coerce")
        pred_df = pred_df.fillna(0)

        return self.cph.predict_median(pred_df)

    def get_cox_summary(self) -> pd.DataFrame:
        """Return Cox model summary (coefficients, hazard ratios, p-values).

        Returns:
            DataFrame with model summary statistics.
        """
        if self.cph is None:
            raise RuntimeError("Cox model not fitted. Call fit_cox_model first.")
        return self.cph.summary

    # ── Plotting ─────────────────────────────────────────────────────────

    def plot_kaplan_meier(self, save_path: str | Path | None = None) -> None:
        """Plot Kaplan-Meier curves (overall and/or by group).

        Args:
            save_path: File path to save the figure. Defaults to
                       docs/diagrams/kaplan_meier.png.
        """
        save_path = Path(save_path) if save_path else DIAGRAM_DIR / "kaplan_meier.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        if self.kmf_groups:
            for label, kmf in self.kmf_groups.items():
                kmf.plot_survival_function(ax=ax)
            ax.set_title("Kaplan-Meier Survival Curves by Contract Type", fontsize=14)
        elif self.kmf:
            self.kmf.plot_survival_function(ax=ax)
            ax.set_title("Kaplan-Meier Survival Curve (Overall)", fontsize=14)
        else:
            ax.text(0.5, 0.5, "No KM model fitted", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)

        ax.set_xlabel("Tenure (months)")
        ax.set_ylabel("Survival Probability")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[SurvivalAnalyzer] Kaplan-Meier plot saved: {save_path}")

    def plot_cox_coefficients(self, save_path: str | Path | None = None) -> None:
        """Plot Cox model coefficients as a forest plot.

        Args:
            save_path: File path to save the figure. Defaults to
                       docs/diagrams/cox_coefficients.png.
        """
        if self.cph is None:
            raise RuntimeError("Cox model not fitted. Call fit_cox_model first.")

        save_path = Path(save_path) if save_path else DIAGRAM_DIR / "cox_coefficients.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        summary = self.cph.summary
        coefs = summary["coef"]
        ci_lower = summary["coef lower 95%"]
        ci_upper = summary["coef upper 95%"]

        fig, ax = plt.subplots(figsize=(8, max(4, len(coefs) * 0.5 + 1)))
        y_pos = range(len(coefs))

        ax.barh(y_pos, coefs.values, color=["#e74c3c" if c > 0 else "#2ecc71" for c in coefs.values],
                alpha=0.7, height=0.6)
        ax.errorbar(coefs.values, y_pos,
                     xerr=[coefs.values - ci_lower.values, ci_upper.values - coefs.values],
                     fmt="none", ecolor="black", capsize=3)
        ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(coefs.index)
        ax.set_xlabel("Coefficient (log hazard ratio)")
        ax.set_title("Cox PH Model Coefficients", fontsize=14)
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[SurvivalAnalyzer] Cox coefficients plot saved: {save_path}")

    # ── Full pipeline ────────────────────────────────────────────────────

    @staticmethod
    def run_full_analysis(df: pd.DataFrame) -> "SurvivalAnalyzer":
        """End-to-end survival analysis: KM overall, KM by contract, Cox PH.

        Generates all diagnostic plots automatically.

        Args:
            df: Processed customer DataFrame with tenure, Churn, and covariates.

        Returns:
            Fitted SurvivalAnalyzer instance.
        """
        analyzer = SurvivalAnalyzer()
        df = SurvivalAnalyzer._prepare_churn_column(df)

        # Ensure tenure ≥ 1
        df["tenure"] = df["tenure"].clip(lower=1)

        # 1. Overall Kaplan-Meier
        print("--- Fitting Kaplan-Meier (overall) ---")
        analyzer.fit_kaplan_meier(df["tenure"], df["Churn"], label="Overall")
        median_overall = analyzer.kmf.median_survival_time_
        print(f"  Median survival time (overall): {median_overall:.1f} months")

        # 2. KM by Contract type
        print("--- Fitting Kaplan-Meier by Contract ---")
        groups = analyzer.fit_kaplan_meier_by_group(df, group_col="Contract")
        for name, kmf in groups.items():
            print(f"  {name}: median survival = {kmf.median_survival_time_:.1f} months")

        # 3. Cox Proportional Hazards
        print("--- Fitting Cox PH model ---")
        analyzer.fit_cox_model(df)
        summary = analyzer.get_cox_summary()
        print("\nCox PH Model Summary:")
        print(summary.to_string())

        # 4. Generate plots
        print("\n--- Generating plots ---")
        analyzer.plot_kaplan_meier()
        analyzer.plot_cox_coefficients()

        return analyzer


# ── CLI entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from config.settings import PROCESSED_DATA_DIR

    csv_path = PROCESSED_DATA_DIR / "telco_churn_processed.csv"
    if not csv_path.exists():
        print(f"ERROR: Processed data not found at {csv_path}")
        print("Run the data pipeline first (scripts/setup_database.py).")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}\n")
    analyzer = SurvivalAnalyzer.run_full_analysis(df)
    print("\n✓ Survival analysis complete.")
