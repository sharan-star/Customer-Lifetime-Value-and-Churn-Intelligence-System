"""
Exploratory Data Analysis Module
==================================
Generates summary statistics and publication-quality visualisations
for the IBM Telco customer-churn dataset.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for cross-package imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import logging
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.settings import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN

logger = logging.getLogger(__name__)

# Consistent palette
sns.set_style("whitegrid")
_CHURN_PALETTE = {"Yes": "#e74c3c", "No": "#2ecc71", 1: "#e74c3c", 0: "#2ecc71"}


class ExploratoryAnalysis:
    """Comprehensive EDA for the Telco churn dataset."""

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Parameters
        ----------
        df : DataFrame
            Processed Telco dataset (may include engineered features).
        """
        self.df: pd.DataFrame = df.copy()
        self.save_dir: Path = _PROJECT_ROOT / "docs" / "diagrams"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Normalise churn column to string for consistent plotting
        if TARGET_COLUMN in self.df.columns and self.df[TARGET_COLUMN].dtype != object:
            self.df[TARGET_COLUMN] = self.df[TARGET_COLUMN].map({1: "Yes", 0: "No"})

    # -------------------------------------------------------- summary stats
    def generate_summary_stats(self) -> Dict[str, Any]:
        """
        High-level summary of the dataset.

        Returns
        -------
        dict
            Keys: shape, dtypes, missing_pct, churn_rate, avg_tenure,
            avg_monthly_charges, avg_total_charges, numerical_stats.
        """
        df = self.df
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        churn_rate: Optional[float] = None
        if TARGET_COLUMN in df.columns:
            churn_col = df[TARGET_COLUMN]
            if churn_col.dtype == object:
                churn_rate = round((churn_col == "Yes").mean(), 4)
            else:
                churn_rate = round(churn_col.mean(), 4)

        stats: Dict[str, Any] = {
            "shape": df.shape,
            "dtypes": df.dtypes.value_counts().to_dict(),
            "missing_values": missing[missing > 0].to_dict(),
            "missing_pct": missing_pct[missing_pct > 0].to_dict(),
            "churn_rate": churn_rate,
            "avg_tenure": round(df["tenure"].mean(), 2) if "tenure" in df.columns else None,
            "avg_monthly_charges": (
                round(df["MonthlyCharges"].mean(), 2) if "MonthlyCharges" in df.columns else None
            ),
            "avg_total_charges": (
                round(df["TotalCharges"].mean(), 2) if "TotalCharges" in df.columns else None
            ),
            "numerical_stats": df.describe().to_dict(),
        }
        logger.info("Summary stats generated – shape=%s, churn_rate=%s", stats["shape"], stats["churn_rate"])
        return stats

    # ======================================================================
    #  PLOTTING METHODS
    # ======================================================================

    def plot_churn_distribution(self, save_path: Optional[str] = None) -> None:
        """Pie chart of churn vs retained."""
        if TARGET_COLUMN not in self.df.columns:
            logger.warning("No '%s' column – skipping churn distribution plot.", TARGET_COLUMN)
            return

        counts = self.df[TARGET_COLUMN].value_counts()
        labels = [f"Retained ({counts.get('No', 0)})", f"Churned ({counts.get('Yes', 0)})"]
        sizes = [counts.get("No", 0), counts.get("Yes", 0)]
        colors = ["#2ecc71", "#e74c3c"]

        fig, ax = plt.subplots(figsize=(7, 7))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"fontsize": 12},
        )
        for t in autotexts:
            t.set_fontweight("bold")
        ax.set_title("Customer Churn Distribution", fontsize=14, fontweight="bold")
        plt.tight_layout()

        dest = Path(save_path) if save_path else self.save_dir / "churn_distribution.png"
        fig.savefig(dest, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Churn distribution plot saved to %s", dest)

    # -------------------------------------------------------- distributions
    def plot_feature_distributions(self, save_path: Optional[str] = None) -> None:
        """Grid of histograms (numerical) and bar-vs-churn charts (categorical)."""
        num_cols = [c for c in NUMERICAL_FEATURES if c in self.df.columns]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in self.df.columns]
        has_churn = TARGET_COLUMN in self.df.columns

        total = len(num_cols) + len(cat_cols)
        if total == 0:
            logger.warning("No features to plot.")
            return

        ncols = 3
        nrows = int(np.ceil(total / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
        axes = np.array(axes).flatten()

        idx = 0
        # Numerical histograms
        for col in num_cols:
            ax = axes[idx]
            if has_churn:
                for label, colour in [("No", "#2ecc71"), ("Yes", "#e74c3c")]:
                    subset = self.df.loc[self.df[TARGET_COLUMN] == label, col].dropna()
                    ax.hist(subset, bins=30, alpha=0.55, label=label, color=colour, edgecolor="white")
                ax.legend(title="Churn")
            else:
                ax.hist(self.df[col].dropna(), bins=30, color="#3498db", edgecolor="white")
            ax.set_title(col)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            idx += 1

        # Categorical bar charts (churn rate)
        for col in cat_cols:
            ax = axes[idx]
            if has_churn:
                ct = pd.crosstab(self.df[col], self.df[TARGET_COLUMN], normalize="index") * 100
                if "Yes" in ct.columns:
                    ct["Yes"].sort_values().plot.barh(ax=ax, color="#e74c3c", edgecolor="white")
                    ax.set_xlabel("Churn Rate (%)")
                else:
                    ct.plot.barh(ax=ax, edgecolor="white")
            else:
                self.df[col].value_counts().plot.barh(ax=ax, color="#3498db", edgecolor="white")
            ax.set_title(f"{col} vs Churn" if has_churn else col)
            idx += 1

        # Hide unused axes
        for j in range(idx, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Feature Distributions", fontsize=16, fontweight="bold", y=1.01)
        plt.tight_layout()

        dest = Path(save_path) if save_path else self.save_dir / "feature_distributions.png"
        fig.savefig(dest, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Feature distributions plot saved to %s", dest)

    # -------------------------------------------------------- correlation
    def plot_correlation_matrix(self, save_path: Optional[str] = None) -> None:
        """Heatmap of correlations among numerical features."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            logger.warning("Not enough numerical columns for correlation matrix.")
            return

        corr = self.df[num_cols].corr()

        fig, ax = plt.subplots(figsize=(max(10, len(num_cols)), max(8, len(num_cols) * 0.7)))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 7},
        )
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()

        dest = Path(save_path) if save_path else self.save_dir / "correlation_matrix.png"
        fig.savefig(dest, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Correlation matrix saved to %s", dest)

    # ------------------------------------------------- churn by feature
    def plot_churn_by_feature(
        self,
        feature: str,
        save_path: Optional[str] = None,
    ) -> None:
        """Grouped bar chart showing churn rate by *feature* categories."""
        if TARGET_COLUMN not in self.df.columns:
            logger.warning("No '%s' column.", TARGET_COLUMN)
            return
        if feature not in self.df.columns:
            logger.warning("Feature '%s' not found.", feature)
            return

        ct = pd.crosstab(self.df[feature], self.df[TARGET_COLUMN])
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ct.plot.bar(ax=ax1, color=["#2ecc71", "#e74c3c"], edgecolor="white")
        ax1.set_title(f"{feature} – Count by Churn Status")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis="x", rotation=45)

        if "Yes" in ct_pct.columns:
            ct_pct["Yes"].plot.bar(ax=ax2, color="#e74c3c", edgecolor="white")
        else:
            ct_pct.plot.bar(ax=ax2, edgecolor="white")
        ax2.set_title(f"{feature} – Churn Rate (%)")
        ax2.set_ylabel("Churn %")
        ax2.tick_params(axis="x", rotation=45)

        plt.suptitle(f"Churn Analysis by {feature}", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        dest = Path(save_path) if save_path else self.save_dir / f"churn_by_{feature.lower()}.png"
        fig.savefig(dest, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Churn-by-%s plot saved to %s", feature, dest)

    # ------------------------------------------------- tenure distribution
    def plot_tenure_distribution(self, save_path: Optional[str] = None) -> None:
        """Tenure histogram split by churn status."""
        if "tenure" not in self.df.columns:
            logger.warning("'tenure' column not found.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        if TARGET_COLUMN in self.df.columns:
            for label, colour in [("No", "#2ecc71"), ("Yes", "#e74c3c")]:
                subset = self.df.loc[self.df[TARGET_COLUMN] == label, "tenure"]
                ax.hist(
                    subset,
                    bins=36,
                    alpha=0.6,
                    label=f"{'Retained' if label == 'No' else 'Churned'} (n={len(subset)})",
                    color=colour,
                    edgecolor="white",
                )
            ax.legend(fontsize=11)
        else:
            ax.hist(self.df["tenure"], bins=36, color="#3498db", edgecolor="white")

        ax.set_title("Tenure Distribution by Churn Status", fontsize=14, fontweight="bold")
        ax.set_xlabel("Tenure (months)")
        ax.set_ylabel("Number of Customers")
        plt.tight_layout()

        dest = Path(save_path) if save_path else self.save_dir / "tenure_distribution.png"
        fig.savefig(dest, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Tenure distribution plot saved to %s", dest)

    # ------------------------------------------ monthly charges vs churn
    def plot_monthly_charges_vs_churn(self, save_path: Optional[str] = None) -> None:
        """Box / violin plot of MonthlyCharges by churn status."""
        if "MonthlyCharges" not in self.df.columns:
            logger.warning("'MonthlyCharges' column not found.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        if TARGET_COLUMN in self.df.columns:
            sns.boxplot(
                data=self.df,
                x=TARGET_COLUMN,
                y="MonthlyCharges",
                palette=_CHURN_PALETTE,
                ax=ax1,
            )
            ax1.set_title("Monthly Charges – Box Plot")

            sns.violinplot(
                data=self.df,
                x=TARGET_COLUMN,
                y="MonthlyCharges",
                palette=_CHURN_PALETTE,
                inner="quartile",
                ax=ax2,
            )
            ax2.set_title("Monthly Charges – Violin Plot")
        else:
            ax1.hist(self.df["MonthlyCharges"], bins=30, color="#3498db", edgecolor="white")
            ax1.set_title("Monthly Charges Distribution")
            ax2.set_visible(False)

        plt.suptitle(
            "Monthly Charges vs Churn", fontsize=14, fontweight="bold", y=1.02
        )
        plt.tight_layout()

        dest = Path(save_path) if save_path else self.save_dir / "monthly_charges_churn.png"
        fig.savefig(dest, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Monthly charges vs churn plot saved to %s", dest)

    # ------------------------------------------------- contract analysis
    def plot_contract_churn_analysis(self, save_path: Optional[str] = None) -> None:
        """Multi-panel contract × churn analysis."""
        if "Contract" not in self.df.columns or TARGET_COLUMN not in self.df.columns:
            logger.warning("Need 'Contract' and '%s' columns.", TARGET_COLUMN)
            return

        ct = pd.crosstab(self.df["Contract"], self.df[TARGET_COLUMN])
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Panel 1 – Counts
        ct.plot.bar(ax=axes[0], color=["#2ecc71", "#e74c3c"], edgecolor="white")
        axes[0].set_title("Churn Count by Contract Type")
        axes[0].set_ylabel("Customer Count")
        axes[0].tick_params(axis="x", rotation=0)

        # Panel 2 – Stacked %
        ct_pct.plot.bar(
            stacked=True,
            ax=axes[1],
            color=["#2ecc71", "#e74c3c"],
            edgecolor="white",
        )
        axes[1].set_title("Churn Proportion by Contract")
        axes[1].set_ylabel("Percentage (%)")
        axes[1].tick_params(axis="x", rotation=0)

        # Panel 3 – Churn rate
        if "Yes" in ct_pct.columns:
            churn_rates = ct_pct["Yes"]
            bars = axes[2].bar(
                churn_rates.index,
                churn_rates.values,
                color=["#e74c3c" if v > 30 else "#f39c12" if v > 15 else "#2ecc71" for v in churn_rates],
                edgecolor="white",
            )
            for bar, val in zip(bars, churn_rates):
                axes[2].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%",
                    ha="center",
                    fontweight="bold",
                )
            axes[2].set_title("Churn Rate (%) by Contract")
            axes[2].set_ylabel("Churn Rate (%)")

        plt.suptitle("Contract Type – Churn Analysis", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        dest = Path(save_path) if save_path else self.save_dir / "contract_analysis.png"
        fig.savefig(dest, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Contract analysis plot saved to %s", dest)

    # ======================================================================
    #  FULL EDA PIPELINE
    # ======================================================================

    def generate_full_eda(self) -> Dict[str, Any]:
        """
        Run every analysis method and generate all plots.

        Returns
        -------
        dict
            Summary statistics plus a list of generated plot paths.
        """
        stats = self.generate_summary_stats()

        plots_generated: List[str] = []

        try:
            self.plot_churn_distribution()
            plots_generated.append("churn_distribution.png")
        except Exception as exc:
            logger.error("plot_churn_distribution failed: %s", exc)

        try:
            self.plot_feature_distributions()
            plots_generated.append("feature_distributions.png")
        except Exception as exc:
            logger.error("plot_feature_distributions failed: %s", exc)

        try:
            self.plot_correlation_matrix()
            plots_generated.append("correlation_matrix.png")
        except Exception as exc:
            logger.error("plot_correlation_matrix failed: %s", exc)

        try:
            self.plot_tenure_distribution()
            plots_generated.append("tenure_distribution.png")
        except Exception as exc:
            logger.error("plot_tenure_distribution failed: %s", exc)

        try:
            self.plot_monthly_charges_vs_churn()
            plots_generated.append("monthly_charges_churn.png")
        except Exception as exc:
            logger.error("plot_monthly_charges_vs_churn failed: %s", exc)

        try:
            self.plot_contract_churn_analysis()
            plots_generated.append("contract_analysis.png")
        except Exception as exc:
            logger.error("plot_contract_churn_analysis failed: %s", exc)

        # Additional churn-by-feature plots for key categorical features
        for feature in ["InternetService", "PaymentMethod", "Contract"]:
            try:
                self.plot_churn_by_feature(feature)
                plots_generated.append(f"churn_by_{feature.lower()}.png")
            except Exception as exc:
                logger.error("plot_churn_by_feature(%s) failed: %s", feature, exc)

        stats["plots_generated"] = plots_generated
        logger.info(
            "Full EDA complete – %d plots generated in %s",
            len(plots_generated),
            self.save_dir,
        )
        return stats
