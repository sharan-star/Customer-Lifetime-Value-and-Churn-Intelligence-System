"""
Uplift Modeling Module for Customer Churn Intervention.

Estimates the causal effect (uplift) of a retention offer on each customer's
churn probability using either causalml (if available) or a manual Two-Model
approach with LightGBM.
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

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from config.settings import PROJECT_ROOT

# Try importing causalml
try:
    from causalml.inference.meta import BaseSClassifier
    _HAS_CAUSALML = True
except ImportError:
    _HAS_CAUSALML = False

# Diagram output directory
DIAGRAM_DIR = PROJECT_ROOT / "docs" / "diagrams"
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)


class UpliftModeler:
    """Estimates heterogeneous treatment effects for retention offers.

    Strategy:
        1. If ``causalml`` is installed, uses ``BaseSClassifier`` with LGBMClassifier.
        2. Otherwise, falls back to a **Two-Model** (T-Learner) approach:
           separate LGBMClassifiers for treatment and control groups.
    """

    def __init__(self) -> None:
        self.treatment_model: LGBMClassifier | None = None
        self.control_model: LGBMClassifier | None = None
        self.causalml_model = None  # BaseSClassifier if available
        self.uplift_scores: np.ndarray | None = None
        self.using_causalml: bool = False

    # ── Fitting ──────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        treatment: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "UpliftModeler":
        """Fit the uplift model.

        Args:
            X: Feature matrix (must NOT contain the treatment column).
            treatment: Binary treatment indicator (1 = treated, 0 = control).
            y: Binary outcome (1 = churned, 0 = retained).

        Returns:
            self for chaining.
        """
        treatment = np.asarray(treatment).ravel()
        y = np.asarray(y).ravel()

        if _HAS_CAUSALML:
            self._fit_causalml(X, treatment, y)
        else:
            self._fit_two_model(X, treatment, y)

        return self

    def _fit_causalml(
        self,
        X: pd.DataFrame | np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Fit using causalml's BaseSClassifier."""
        print("[UpliftModeler] Using causalml BaseSClassifier")
        self.using_causalml = True
        learner = LGBMClassifier(
            n_estimators=200, learning_rate=0.05,
            num_leaves=31, verbose=-1, random_state=42,
        )
        # Keep the treatment labels explicit and configure the matching
        # control level; CausalML defaults to numeric control label ``0``.
        self.causalml_model = BaseSClassifier(learner=learner, control_name="control")
        treatment_str = np.where(treatment == 1, "treatment", "control")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.causalml_model.fit(X, treatment_str, y)

    def _fit_two_model(
        self,
        X: pd.DataFrame | np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Fit using the Two-Model (T-Learner) approach."""
        print("[UpliftModeler] causalml not available — using Two-Model approach")
        self.using_causalml = False

        treat_mask = treatment == 1
        ctrl_mask = treatment == 0

        self.treatment_model = LGBMClassifier(
            n_estimators=200, learning_rate=0.05,
            num_leaves=31, verbose=-1, random_state=42,
        )
        self.control_model = LGBMClassifier(
            n_estimators=200, learning_rate=0.05,
            num_leaves=31, verbose=-1, random_state=42,
        )

        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.treatment_model.fit(X_arr[treat_mask], y[treat_mask])
            self.control_model.fit(X_arr[ctrl_mask], y[ctrl_mask])

    # ── Prediction ───────────────────────────────────────────────────────

    def predict_uplift(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict individual-level uplift scores.

        Positive uplift means the treatment **reduces** churn probability
        (the customer is *persuadable*).

        Args:
            X: Feature matrix (same shape as training features).

        Returns:
            1-D array of uplift scores.
        """
        if self.using_causalml and self.causalml_model is not None:
            # causalml returns (n, 1) or dict
            preds = self.causalml_model.predict(X)
            if isinstance(preds, dict):
                preds = list(preds.values())[0]
            self.uplift_scores = np.asarray(preds).ravel()
        else:
            X_arr = X.values if isinstance(X, pd.DataFrame) else X
            p_churn_control = self.control_model.predict_proba(X_arr)[:, 1]
            p_churn_treatment = self.treatment_model.predict_proba(X_arr)[:, 1]
            # Uplift = P(churn|control) - P(churn|treatment)
            # Positive → treatment helps (reduces churn)
            self.uplift_scores = p_churn_control - p_churn_treatment

        return self.uplift_scores

    # ── Segmentation ─────────────────────────────────────────────────────

    def get_uplift_segments(
        self,
        X: pd.DataFrame | np.ndarray,
        treatment: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> pd.DataFrame:
        """Segment customers by uplift response type.

        Segments:
            - **Persuadable**: top-quartile positive uplift → target these.
            - **Sure Thing**: low churn risk regardless of treatment.
            - **Lost Cause**: high churn risk regardless of treatment.
            - **Sleeping Dog**: negative uplift → treatment *increases* churn!

        Args:
            X: Feature matrix.
            treatment: Treatment indicator.
            y: Churn outcome.

        Returns:
            DataFrame with ``uplift_score`` and ``segment`` columns.
        """
        treatment = np.asarray(treatment).ravel()
        y = np.asarray(y).ravel()

        if self.uplift_scores is None:
            self.predict_uplift(X)

        scores = self.uplift_scores

        # Compute base churn probabilities for segmentation
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        if self.using_causalml:
            # Estimate base churn from control model (not available via causalml)
            # Fall back to actual outcome
            base_churn_prob = y.astype(float)
        else:
            base_churn_prob = self.control_model.predict_proba(X_arr)[:, 1]

        # Thresholds
        uplift_q75 = np.percentile(scores, 75)
        churn_median = np.median(base_churn_prob)

        segments = []
        for i in range(len(scores)):
            if scores[i] < 0:
                segments.append("Sleeping Dog")
            elif scores[i] >= uplift_q75:
                segments.append("Persuadable")
            elif base_churn_prob[i] < churn_median:
                segments.append("Sure Thing")
            else:
                segments.append("Lost Cause")

        result = pd.DataFrame({
            "uplift_score": scores,
            "segment": segments,
            "treatment": treatment,
            "churn": y,
        })
        if isinstance(X, pd.DataFrame):
            result.index = X.index
        return result

    # ── Plotting ─────────────────────────────────────────────────────────

    def plot_uplift_distribution(self, save_path: str | Path | None = None) -> None:
        """Plot histogram of uplift scores.

        Args:
            save_path: Defaults to docs/diagrams/uplift_distribution.png.
        """
        if self.uplift_scores is None:
            raise RuntimeError("No uplift scores. Call predict_uplift first.")

        save_path = Path(save_path) if save_path else DIAGRAM_DIR / "uplift_distribution.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(self.uplift_scores, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1.2, label="Zero uplift")
        ax.axvline(x=np.median(self.uplift_scores), color="green", linestyle="--",
                   linewidth=1.2, label=f"Median ({np.median(self.uplift_scores):.3f})")
        ax.set_xlabel("Uplift Score (ΔP churn)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Individual Uplift Scores", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[UpliftModeler] Uplift distribution plot saved: {save_path}")

    def plot_uplift_by_segment(
        self,
        segments_df: pd.DataFrame,
        save_path: str | Path | None = None,
    ) -> None:
        """Bar chart: average churn rate per segment for treatment vs control.

        Args:
            segments_df: Output of get_uplift_segments().
            save_path: Defaults to docs/diagrams/uplift_segments.png.
        """
        save_path = Path(save_path) if save_path else DIAGRAM_DIR / "uplift_segments.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        segment_order = ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"]
        summary = (
            segments_df.groupby(["segment", "treatment"])["churn"]
            .mean()
            .unstack(fill_value=0)
        )
        # Reindex to desired order (drop any missing segments)
        summary = summary.reindex([s for s in segment_order if s in summary.index])

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(summary))
        width = 0.35

        bars_ctrl = ax.bar(x - width / 2, summary.get(0, 0), width,
                           label="Control (no offer)", color="#e74c3c", alpha=0.8)
        bars_treat = ax.bar(x + width / 2, summary.get(1, 0), width,
                            label="Treatment (offer)", color="#2ecc71", alpha=0.8)

        ax.set_xlabel("Customer Segment")
        ax.set_ylabel("Churn Rate")
        ax.set_title("Churn Rate by Uplift Segment: Treatment vs Control", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(summary.index, rotation=15)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

        # Add value labels
        for bar_group in [bars_ctrl, bars_treat]:
            for bar in bar_group:
                h = bar.get_height()
                if h > 0:
                    ax.annotate(f"{h:.1%}", xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 4), textcoords="offset points",
                                ha="center", fontsize=9)

        plt.tight_layout()
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[UpliftModeler] Uplift segments plot saved: {save_path}")

    def plot_qini_curve(
        self,
        X: pd.DataFrame | np.ndarray,
        treatment: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
        save_path: str | Path | None = None,
    ) -> None:
        """Plot Qini (cumulative uplift) curve.

        Sorts customers by predicted uplift (descending) and plots the
        cumulative incremental gain from targeting them.

        Args:
            X: Feature matrix.
            treatment: Treatment indicator.
            y: Churn outcome.
            save_path: Defaults to docs/diagrams/qini_curve.png.
        """
        save_path = Path(save_path) if save_path else DIAGRAM_DIR / "qini_curve.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        treatment = np.asarray(treatment).ravel()
        y = np.asarray(y).ravel()

        if self.uplift_scores is None:
            self.predict_uplift(X)

        # Sort by uplift descending
        order = np.argsort(-self.uplift_scores)
        sorted_treatment = treatment[order]
        sorted_y = y[order]

        n = len(sorted_y)
        n_treat = (sorted_treatment == 1).sum()
        n_ctrl = (sorted_treatment == 0).sum()

        if n_treat == 0 or n_ctrl == 0:
            print("[UpliftModeler] Warning: cannot plot Qini — no treatment/control split")
            return

        cumulative_uplift = []
        cum_treat_churn = 0
        cum_ctrl_churn = 0
        cum_treat_n = 0
        cum_ctrl_n = 0

        for i in range(n):
            if sorted_treatment[i] == 1:
                cum_treat_churn += sorted_y[i]
                cum_treat_n += 1
            else:
                cum_ctrl_churn += sorted_y[i]
                cum_ctrl_n += 1

            # Incremental gain: churn prevented per customer targeted
            treat_rate = cum_treat_churn / cum_treat_n if cum_treat_n > 0 else 0
            ctrl_rate = cum_ctrl_churn / cum_ctrl_n if cum_ctrl_n > 0 else 0
            lift = (ctrl_rate - treat_rate) * (i + 1) / n
            cumulative_uplift.append(lift)

        fractions = np.linspace(0, 1, n + 1)[1:]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(fractions, cumulative_uplift, color="#3498db", linewidth=2, label="Model")
        ax.plot([0, 1], [0, cumulative_uplift[-1]], color="gray", linestyle="--",
                linewidth=1, label="Random")
        ax.fill_between(fractions, cumulative_uplift,
                        np.linspace(0, cumulative_uplift[-1], n),
                        alpha=0.15, color="#3498db")
        ax.set_xlabel("Fraction of Population Targeted")
        ax.set_ylabel("Cumulative Uplift (Churn Prevented)")
        ax.set_title("Qini Curve — Cumulative Uplift", fontsize=14)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[UpliftModeler] Qini curve saved: {save_path}")

    # ── Full pipeline ────────────────────────────────────────────────────

    @staticmethod
    def run_full_analysis(df: pd.DataFrame) -> tuple["UpliftModeler", pd.DataFrame]:
        """End-to-end uplift analysis pipeline.

        Prepares features, fits the uplift model, segments customers,
        and generates all diagnostic plots.

        Args:
            df: Processed customer DataFrame with a ``treatment`` column.

        Returns:
            Tuple of (fitted UpliftModeler, segments DataFrame).
        """
        from src.data_pipeline.feature_engineering import prepare_model_data

        # Prepare model data
        X, y = prepare_model_data(df)

        # Remove treatment from features if present
        if "treatment" in X.columns:
            treatment = X["treatment"].copy()
            X = X.drop(columns=["treatment"])
        elif "treatment" in df.columns:
            treatment = df["treatment"].copy()
            # Align index
            treatment = treatment.loc[X.index]
        else:
            raise ValueError("No 'treatment' column found in the data.")

        treatment = treatment.astype(int)
        y = y.astype(int)

        print(f"[UpliftModeler] Data shape: X={X.shape}, treatment distribution: "
              f"treated={int((treatment == 1).sum())}, control={int((treatment == 0).sum())}")
        print(f"[UpliftModeler] Using causalml: {_HAS_CAUSALML}")

        # Fit
        modeler = UpliftModeler()
        modeler.fit(X, treatment, y)

        # Predict uplift
        uplift = modeler.predict_uplift(X)
        print(f"\nUplift score stats:")
        print(f"  Mean:   {np.mean(uplift):.4f}")
        print(f"  Median: {np.median(uplift):.4f}")
        print(f"  Min:    {np.min(uplift):.4f}")
        print(f"  Max:    {np.max(uplift):.4f}")

        # Segments
        segments_df = modeler.get_uplift_segments(X, treatment, y)
        print(f"\nUplift segments:")
        print(segments_df["segment"].value_counts().to_string())

        # Churn rate by segment × treatment
        print(f"\nChurn rate by segment and treatment:")
        crosstab = segments_df.groupby(["segment", "treatment"])["churn"].mean()
        print(crosstab.to_string())

        # Plots
        print("\n--- Generating plots ---")
        modeler.plot_uplift_distribution()
        modeler.plot_uplift_by_segment(segments_df)
        modeler.plot_qini_curve(X, treatment, y)

        return modeler, segments_df


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
    modeler, segments = UpliftModeler.run_full_analysis(df)
    print("\n✓ Uplift analysis complete.")
