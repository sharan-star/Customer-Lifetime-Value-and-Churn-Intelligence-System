"""
SHAP Explainer Module
======================
Model-agnostic SHAP explanations for the churn classifier, with
publication-ready plots.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import logging
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must come before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

logger = logging.getLogger(__name__)

# Default output directory for plots
_DEFAULT_DIAGRAM_DIR = _PROJECT_ROOT / "docs" / "diagrams"


class ShapExplainer:
    """Compute and visualise SHAP values for a trained LightGBM model."""

    def __init__(self, model, X_data: pd.DataFrame) -> None:
        """
        Create a SHAP TreeExplainer and compute SHAP values.

        Parameters
        ----------
        model : lgb.LGBMClassifier
            A trained LightGBM classifier (the *sklearn* API wrapper).
        X_data : pd.DataFrame
            Feature matrix to explain.
        """
        self.model = model
        self.X_data: pd.DataFrame = X_data.copy()
        self.feature_names: List[str] = list(X_data.columns)

        # Build explainer
        self.explainer = shap.TreeExplainer(model)

        # Compute SHAP values — handle both old (array) and new (Explanation) API
        raw = self.explainer.shap_values(X_data)

        if isinstance(raw, list):
            # Binary classification: list of [class-0, class-1] arrays
            self.shap_values: np.ndarray = np.array(raw[1])
        elif isinstance(raw, np.ndarray) and raw.ndim == 3:
            self.shap_values = raw[:, :, 1]
        else:
            self.shap_values = np.array(raw)

        # Build an Explanation object for newer SHAP plots
        self._explanation = shap.Explanation(
            values=self.shap_values,
            base_values=np.full(self.shap_values.shape[0], self._get_base_value()),
            data=self.X_data.values,
            feature_names=self.feature_names,
        )

        logger.info(
            "SHAP values computed for %d samples × %d features",
            *self.shap_values.shape,
        )

    # ---------------------------------------------------------- accessors
    def get_shap_values(self) -> np.ndarray:
        """Return the SHAP values array (samples × features)."""
        return self.shap_values

    def get_feature_importance_shap(self) -> pd.DataFrame:
        """
        Return a DataFrame of features sorted by mean |SHAP value|.

        Columns: ``feature``, ``mean_abs_shap``.
        """
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        df = pd.DataFrame(
            {"feature": self.feature_names, "mean_abs_shap": mean_abs}
        )
        return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def get_top_features(self, n: int = 10) -> List[str]:
        """Return the top *n* feature names by mean |SHAP value|."""
        importance = self.get_feature_importance_shap()
        return importance["feature"].head(n).tolist()

    # ------------------------------------------------------------- plots
    def plot_summary(self, save_path: Optional[str] = None) -> None:
        """
        Generate a SHAP beeswarm (summary) plot.

        Parameters
        ----------
        save_path : str, optional
            File path for the saved figure.  Defaults to
            ``docs/diagrams/shap_summary.png``.
        """
        out = Path(save_path) if save_path else _DEFAULT_DIAGRAM_DIR / "shap_summary.png"
        out.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_data,
            feature_names=self.feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Summary plot saved to %s", out)

    def plot_waterfall(self, idx: int = 0, save_path: Optional[str] = None) -> None:
        """
        Generate a SHAP waterfall plot for a single observation.

        Parameters
        ----------
        idx : int
            Row index in ``X_data`` to explain.
        save_path : str, optional
            Defaults to ``docs/diagrams/shap_waterfall.png``.
        """
        out = Path(save_path) if save_path else _DEFAULT_DIAGRAM_DIR / "shap_waterfall.png"
        out.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 8))
        try:
            shap.plots.waterfall(self._explanation[idx], show=False)
        except (AttributeError, TypeError):
            # Fallback for older SHAP versions
            shap.waterfall_plot(
                shap.Explanation(
                    values=self.shap_values[idx],
                    base_values=self._get_base_value(),
                    data=self.X_data.iloc[idx].values,
                    feature_names=self.feature_names,
                ),
                show=False,
            )
        plt.tight_layout()
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Waterfall plot saved to %s", out)

    def plot_bar(self, save_path: Optional[str] = None) -> None:
        """
        Generate a bar plot of mean |SHAP| values.

        Parameters
        ----------
        save_path : str, optional
            Defaults to ``docs/diagrams/shap_bar.png``.
        """
        out = Path(save_path) if save_path else _DEFAULT_DIAGRAM_DIR / "shap_bar.png"
        out.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 8))
        try:
            shap.plots.bar(self._explanation, show=False)
        except (AttributeError, TypeError):
            # Fallback: manual bar chart
            importance = self.get_feature_importance_shap().head(20)
            plt.barh(importance["feature"][::-1], importance["mean_abs_shap"][::-1])
            plt.xlabel("Mean |SHAP value|")
            plt.title("Feature Importance (SHAP)")
        plt.tight_layout()
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Bar plot saved to %s", out)

    # ------------------------------------------------ convenience static
    @staticmethod
    def generate_explanations(
        classifier,
        X_test: pd.DataFrame,
        save_dir: Optional[str] = None,
    ) -> "ShapExplainer":
        """
        One-call convenience: create explainer, generate all plots.

        Parameters
        ----------
        classifier : ChurnClassifier
            A trained ``ChurnClassifier`` instance.
        X_test : pd.DataFrame
            Feature matrix to explain (typically the test set).
        save_dir : str, optional
            Directory for plot outputs.  Defaults to ``docs/diagrams/``.

        Returns
        -------
        ShapExplainer
            The fully-initialised explainer with computed SHAP values.
        """
        explainer = ShapExplainer(classifier.model, X_test)

        # Resolve save paths
        base = Path(save_dir) if save_dir else _DEFAULT_DIAGRAM_DIR
        base.mkdir(parents=True, exist_ok=True)

        explainer.plot_summary(save_path=str(base / "shap_summary.png"))
        explainer.plot_waterfall(idx=0, save_path=str(base / "shap_waterfall.png"))
        explainer.plot_bar(save_path=str(base / "shap_bar.png"))

        logger.info("All SHAP explanation plots generated in %s", base)
        return explainer

    # ---------------------------------------------------------- internal
    def _get_base_value(self) -> float:
        """Extract the base (expected) value for the positive class."""
        ev = self.explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            return float(ev[1]) if len(ev) > 1 else float(ev[0])
        return float(ev)
