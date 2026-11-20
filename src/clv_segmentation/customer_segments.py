"""
Customer Segmentation Module
==============================
KMeans-based customer segmentation with automatic labelling,
profile generation, and visualisation.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for cross-package imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config.settings import N_CLUSTERS

logger = logging.getLogger(__name__)

_DIAGRAM_DIR = _PROJECT_ROOT / "docs" / "diagrams"

# Default feature set for segmentation
_DEFAULT_FEATURES: List[str] = [
    "MonthlyCharges",
    "tenure",
    "clv",
    "num_services",
    "churn_probability",
]


class CustomerSegmenter:
    """KMeans-based customer segmenter with profiling and visualisation."""

    def __init__(self, n_clusters: Optional[int] = None) -> None:
        """
        Initialise the segmenter.

        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters.  Defaults to ``N_CLUSTERS`` from settings.
        """
        self.n_clusters: int = n_clusters if n_clusters is not None else N_CLUSTERS
        self.scaler = StandardScaler()
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
        )
        self.features_used: Optional[List[str]] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------ fit
    def fit_segments(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> "CustomerSegmenter":
        """
        Scale features and fit the KMeans model.

        Parameters
        ----------
        df : DataFrame
            Customer-level data.
        features : list[str], optional
            Columns to use.  Falls back to ``_DEFAULT_FEATURES``.

        Returns
        -------
        CustomerSegmenter
            *self*, for method chaining.
        """
        self.features_used = self._resolve_features(df, features)
        X = df[self.features_used].copy()
        X.fillna(X.median(), inplace=True)
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)
        self._is_fitted = True
        logger.info(
            "KMeans fitted – k=%d, inertia=%.2f, features=%s",
            self.n_clusters,
            self.kmeans.inertia_,
            self.features_used,
        )
        return self

    # -------------------------------------------------------------- predict
    def predict_segments(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Assign cluster labels to *df*.

        Parameters
        ----------
        df : DataFrame
        features : list[str], optional

        Returns
        -------
        np.ndarray
            Integer cluster labels.
        """
        if not self._is_fitted:
            raise RuntimeError("Segmenter has not been fitted – call fit_segments first.")
        feats = features if features else self.features_used
        X = df[feats].copy()
        X.fillna(X.median(), inplace=True)
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)

    # ----------------------------------------------------------- profiles
    def get_segment_profiles(
        self,
        df: pd.DataFrame,
        segment_col: str = "segment",
    ) -> pd.DataFrame:
        """
        Compute descriptive profiles for each segment.

        Parameters
        ----------
        df : DataFrame
            Must contain *segment_col*.  Columns such as ``MonthlyCharges``,
            ``tenure``, ``clv``, ``churn_probability``, ``Contract``, and
            ``Churn`` are used when present.
        segment_col : str
            Column holding cluster labels.

        Returns
        -------
        pd.DataFrame
            One row per segment with aggregate statistics.
        """
        profiles: List[Dict] = []
        total = len(df)

        for seg_id in sorted(df[segment_col].unique()):
            mask = df[segment_col] == seg_id
            seg = df.loc[mask]
            row: Dict = {
                "segment": seg_id,
                "count": len(seg),
                "pct_of_total": round(len(seg) / total * 100, 1),
            }

            for col in ["MonthlyCharges", "tenure", "clv", "TotalCharges"]:
                if col in seg.columns:
                    row[f"avg_{col}"] = round(seg[col].mean(), 2)

            if "churn_probability" in seg.columns:
                row["avg_churn_probability"] = round(seg["churn_probability"].mean(), 4)

            if "Churn" in seg.columns:
                churn_col = seg["Churn"]
                if churn_col.dtype == object:
                    churn_col = churn_col.map({"Yes": 1, "No": 0})
                row["churn_rate"] = round(churn_col.mean(), 4)

            if "Contract" in seg.columns:
                row["dominant_contract"] = seg["Contract"].mode().iloc[0] if len(seg) > 0 else "N/A"

            if "num_services" in seg.columns:
                row["avg_num_services"] = round(seg["num_services"].mean(), 2)

            # Recommended action heuristic
            avg_clv = row.get("avg_clv", 0)
            avg_churn = row.get("avg_churn_probability", row.get("churn_rate", 0))
            row["recommended_action"] = self._recommend_action(avg_clv, avg_churn, df)

            profiles.append(row)

        return pd.DataFrame(profiles)

    # -------------------------------------------------------------- labels
    def label_segments(self, profiles: pd.DataFrame) -> Dict[int, str]:
        """
        Auto-label segments based on CLV and churn characteristics.

        Labels
        ------
        - **Champions**: highest CLV + lowest churn
        - **At Risk High Value**: high CLV + high churn
        - **Loyal Low Value**: low CLV + low churn
        - **Price Sensitive**: low CLV + high churn

        Returns
        -------
        dict
            ``{cluster_id: label}``
        """
        churn_col = "avg_churn_probability" if "avg_churn_probability" in profiles.columns else "churn_rate"
        if "avg_clv" not in profiles.columns or churn_col not in profiles.columns:
            return {row["segment"]: f"Segment {row['segment']}" for _, row in profiles.iterrows()}

        median_clv = profiles["avg_clv"].median()
        median_churn = profiles[churn_col].median()

        labels: Dict[int, str] = {}
        for _, row in profiles.iterrows():
            high_clv = row["avg_clv"] >= median_clv
            high_churn = row[churn_col] >= median_churn
            if high_clv and not high_churn:
                labels[row["segment"]] = "Champions"
            elif high_clv and high_churn:
                labels[row["segment"]] = "At Risk High Value"
            elif not high_clv and not high_churn:
                labels[row["segment"]] = "Loyal Low Value"
            else:
                labels[row["segment"]] = "Price Sensitive"

        return labels

    # -------------------------------------------------------------- plots
    def plot_segments(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> None:
        """
        2-D scatter of customers coloured by segment (PCA if >2 features).

        Saved to ``docs/diagrams/customer_segments.png``.
        """
        feats = self.features_used or _DEFAULT_FEATURES
        available = [f for f in feats if f in df.columns]
        X = df[available].copy()
        X.fillna(X.median(), inplace=True)
        X_scaled = self.scaler.transform(X)

        if X_scaled.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_scaled)
            x_label = f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)"
            y_label = f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)"
        else:
            coords = X_scaled
            x_label, y_label = available[0], available[1]

        seg_col = df["segment"] if "segment" in df.columns else self.predict_segments(df)

        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=seg_col,
            cmap="Set2",
            alpha=0.6,
            s=15,
            edgecolors="none",
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title("Customer Segments (KMeans)")
        plt.colorbar(scatter, ax=ax, label="Segment")
        plt.tight_layout()

        dest = Path(save_path) if save_path else _DIAGRAM_DIR / "customer_segments.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Segment scatter saved to %s", dest)

    def plot_segment_profiles(
        self,
        profiles: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Grouped bar chart of key metrics by segment.

        Saved to ``docs/diagrams/segment_profiles.png``.
        """
        metric_cols = [
            c for c in profiles.columns
            if c.startswith("avg_") or c in ("churn_rate", "pct_of_total")
        ]
        if not metric_cols:
            logger.warning("No numeric profile columns to plot.")
            return

        labels_map = self.label_segments(profiles)
        display_labels = [
            labels_map.get(seg, f"Seg {seg}") for seg in profiles["segment"]
        ]

        fig, axes = plt.subplots(1, len(metric_cols), figsize=(5 * len(metric_cols), 5))
        if len(metric_cols) == 1:
            axes = [axes]

        palette = sns.color_palette("Set2", len(profiles))
        for ax, col in zip(axes, metric_cols):
            bars = ax.bar(display_labels, profiles[col], color=palette)
            ax.set_title(col.replace("avg_", "Avg ").replace("_", " ").title())
            ax.set_ylabel(col)
            ax.tick_params(axis="x", rotation=30)
            for bar, val in zip(bars, profiles[col]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.suptitle("Segment Profiles", fontsize=14, y=1.02)
        plt.tight_layout()

        dest = Path(save_path) if save_path else _DIAGRAM_DIR / "segment_profiles.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Segment profiles chart saved to %s", dest)

    # -------------------------------------------------- end-to-end
    @staticmethod
    def run_segmentation(
        df: pd.DataFrame,
    ) -> Tuple["CustomerSegmenter", pd.DataFrame, pd.DataFrame]:
        """
        End-to-end segmentation pipeline.

        1. Fit KMeans on the default feature set.
        2. Assign segment labels.
        3. Generate profiles and auto-labels.
        4. Save visualisations.

        Returns
        -------
        tuple
            ``(segmenter, segments_df, profiles)``
        """
        segmenter = CustomerSegmenter()
        segmenter.fit_segments(df)
        segments = segmenter.predict_segments(df)

        segments_df = df.copy()
        segments_df["segment"] = segments

        profiles = segmenter.get_segment_profiles(segments_df)
        labels = segmenter.label_segments(profiles)
        profiles["label"] = profiles["segment"].map(labels)
        segments_df["segment_label"] = segments_df["segment"].map(labels)

        segmenter.plot_segments(segments_df)
        segmenter.plot_segment_profiles(profiles)

        logger.info("Segmentation complete – profiles:\n%s", profiles.to_string())
        return segmenter, segments_df, profiles

    # --------------------------------------------------------- internal
    @staticmethod
    def _resolve_features(
        df: pd.DataFrame, features: Optional[List[str]]
    ) -> List[str]:
        """Return the intersection of requested features and available columns."""
        candidates = features if features else _DEFAULT_FEATURES
        available = [f for f in candidates if f in df.columns]
        if not available:
            raise ValueError(
                f"None of the requested features {candidates} found in DataFrame "
                f"(columns: {list(df.columns[:20])}…)"
            )
        return available

    @staticmethod
    def _recommend_action(avg_clv: float, avg_churn: float, df: pd.DataFrame) -> str:
        """Simple heuristic for segment-level recommended action."""
        global_median_clv = df["clv"].median() if "clv" in df.columns else avg_clv
        high_clv = avg_clv >= global_median_clv
        high_churn = avg_churn >= 0.35

        if high_clv and high_churn:
            return "Urgent retention – personal outreach & premium offers"
        if high_clv and not high_churn:
            return "Loyalty programme – upsell & cross-sell"
        if not high_clv and high_churn:
            return "Value-based discounts – cost-sensitive retention"
        return "Maintain service quality – monitor satisfaction"
