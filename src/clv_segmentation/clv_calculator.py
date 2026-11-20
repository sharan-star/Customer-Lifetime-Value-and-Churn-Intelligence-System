"""
CLV Calculator Module
======================
Computes Customer Lifetime Value using discounted cash-flow models.
Supports simple geometric-approximation CLV, survival-curve CLV,
and expected revenue-impact analysis for retention campaigns.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for cross-package imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import AVERAGE_MONTHLY_MARGIN, DISCOUNT_RATE_ANNUAL

logger = logging.getLogger(__name__)


class CLVCalculator:
    """Discounted cash-flow customer lifetime value calculator."""

    def __init__(
        self,
        discount_rate: Optional[float] = None,
        margin_rate: Optional[float] = None,
    ) -> None:
        """
        Initialise the calculator.

        Parameters
        ----------
        discount_rate : float, optional
            Annual discount rate.  Defaults to ``DISCOUNT_RATE_ANNUAL``
            from *config/settings.py*.
        margin_rate : float, optional
            Monthly margin as a fraction of ``MonthlyCharges``.
            Defaults to ``AVERAGE_MONTHLY_MARGIN`` from settings.
        """
        annual_rate = discount_rate if discount_rate is not None else DISCOUNT_RATE_ANNUAL
        self.margin_rate: float = margin_rate if margin_rate is not None else AVERAGE_MONTHLY_MARGIN
        # Convert annual discount rate to monthly
        self.monthly_discount: float = (1 + annual_rate) ** (1 / 12) - 1
        logger.info(
            "CLVCalculator initialised – monthly discount=%.6f, margin=%.2f",
            self.monthly_discount,
            self.margin_rate,
        )

    # ------------------------------------------------------------------ core
    def compute_simple_clv(
        self, monthly_charges: float, expected_tenure_months: int
    ) -> float:
        """
        Compute CLV as the discounted sum of future monthly margins.

        CLV = Σ_{t=1}^{T}  (margin × monthly_charge) / (1 + r)^t

        Parameters
        ----------
        monthly_charges : float
            Customer's monthly bill.
        expected_tenure_months : int
            Number of months the customer is expected to remain.

        Returns
        -------
        float
            Present-value lifetime value.
        """
        r = self.monthly_discount
        margin = self.margin_rate
        clv = 0.0
        for t in range(1, int(expected_tenure_months) + 1):
            clv += (margin * monthly_charges) / ((1 + r) ** t)
        return clv

    # ----------------------------------------------------------- DataFrame
    def compute_clv_for_dataframe(
        self,
        df: pd.DataFrame,
        churn_prob_col: str = "churn_probability",
        monthly_charge_col: str = "MonthlyCharges",
    ) -> pd.Series:
        """
        Vectorised CLV for every customer in *df*.

        Expected remaining tenure is estimated from the churn probability
        using the geometric-distribution approximation::

            E[T] = (1 - p) / p

        Capped at 72 months, floored at 1 month.

        Parameters
        ----------
        df : DataFrame
            Must contain *churn_prob_col* and *monthly_charge_col*.
        churn_prob_col : str
            Column with churn probabilities in [0, 1].
        monthly_charge_col : str
            Column with monthly charges.

        Returns
        -------
        pd.Series
            CLV value for each row, index-aligned to *df*.
        """
        p = df[churn_prob_col].clip(lower=0.001, upper=0.999)
        expected_tenure = ((1 - p) / p).clip(lower=1, upper=72).round().astype(int)
        charges = df[monthly_charge_col].values

        clv_values = np.array(
            [
                self.compute_simple_clv(charge, tenure)
                for charge, tenure in zip(charges, expected_tenure)
            ]
        )
        return pd.Series(clv_values, index=df.index, name="clv")

    # ------------------------------------------------------- survival CLV
    def compute_clv_with_survival(
        self,
        df: pd.DataFrame,
        survival_probs: pd.DataFrame,
        monthly_charge_col: str = "MonthlyCharges",
    ) -> pd.Series:
        """
        CLV using per-period survival probabilities (e.g. from a Cox model).

        CLV_i = Σ_t  S_i(t) × margin × charge_i / (1 + r)^t

        Parameters
        ----------
        df : DataFrame
            Customer-level data with *monthly_charge_col*.
        survival_probs : DataFrame
            Rows = customers (aligned to *df*), columns = time periods
            (1, 2, …, T).  Values are P(survive ≥ t).
        monthly_charge_col : str
            Column with monthly charges.

        Returns
        -------
        pd.Series
            CLV value for each customer.
        """
        r = self.monthly_discount
        margin = self.margin_rate
        charges = df[monthly_charge_col].values  # shape (n,)

        # Discount factors for each period column
        periods = np.arange(1, survival_probs.shape[1] + 1)
        discount_factors = 1 / ((1 + r) ** periods)  # shape (T,)

        # S(t) matrix * discount * margin * charge
        surv = survival_probs.values  # (n, T)
        clv_matrix = surv * discount_factors[np.newaxis, :] * margin * charges[:, np.newaxis]
        clv_values = clv_matrix.sum(axis=1)

        return pd.Series(clv_values, index=df.index, name="clv_survival")

    # -------------------------------------------------- revenue-impact
    def compute_expected_revenue_impact(
        self,
        df: pd.DataFrame,
        uplift_scores: np.ndarray,
        offer_cost: float,
        monthly_charge_col: str = "MonthlyCharges",
    ) -> pd.DataFrame:
        """
        Estimate the revenue impact of a retention offer for each customer.

        Parameters
        ----------
        df : DataFrame
            Customer data with *monthly_charge_col* and ``churn_probability``.
        uplift_scores : array-like
            Predicted uplift (increase in retention probability) per customer.
        offer_cost : float
            Per-customer cost of the retention offer.
        monthly_charge_col : str
            Column with monthly charges.

        Returns
        -------
        pd.DataFrame
            Columns: ``expected_retention_months_gained``,
            ``revenue_from_retention``, ``roi``, ``net_value``.
        """
        r = self.monthly_discount
        margin = self.margin_rate
        charges = df[monthly_charge_col].values
        uplift = np.asarray(uplift_scores).clip(min=0)

        # Baseline expected remaining tenure (geometric approx)
        p = df["churn_probability"].clip(lower=0.001, upper=0.999).values
        base_tenure = ((1 - p) / p).clip(min=1, max=72)

        months_gained = uplift * base_tenure

        # Discounted revenue from the additional months
        revenue = np.array(
            [
                sum(
                    (margin * charge) / ((1 + r) ** t)
                    for t in range(1, int(round(mg)) + 1)
                )
                if mg >= 0.5
                else 0.0
                for charge, mg in zip(charges, months_gained)
            ]
        )

        roi = np.where(offer_cost > 0, (revenue - offer_cost) / offer_cost, 0.0)
        net_value = revenue - offer_cost

        return pd.DataFrame(
            {
                "expected_retention_months_gained": months_gained,
                "revenue_from_retention": revenue,
                "roi": roi,
                "net_value": net_value,
            },
            index=df.index,
        )

    # ---------------------------------------------------- convenience
    @staticmethod
    def compute_all_clv(
        df: pd.DataFrame,
        churn_probs: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        End-to-end convenience: attach a ``clv`` column to *df*.

        If *churn_probs* are not supplied, they are estimated from the
        overall churn rate within each tenure group (``tenure_group`` column
        if present, else quartile bins of ``tenure``).

        Parameters
        ----------
        df : DataFrame
            Must contain ``MonthlyCharges`` and ``tenure``.
        churn_probs : array-like, optional
            Per-customer churn probabilities.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with an added ``clv`` column.
        """
        out = df.copy()
        calc = CLVCalculator()

        if churn_probs is not None:
            out["churn_probability"] = churn_probs
        elif "churn_probability" not in out.columns:
            # Estimate from average churn rate by tenure group
            if "tenure_group" in out.columns:
                group_col = "tenure_group"
            else:
                out["_tenure_q"] = pd.qcut(out["tenure"], q=4, duplicates="drop")
                group_col = "_tenure_q"

            if "Churn" in out.columns:
                churn_col = out["Churn"]
                if churn_col.dtype == object:
                    churn_col = churn_col.map({"Yes": 1, "No": 0})
                group_rate = churn_col.groupby(out[group_col]).transform("mean")
            else:
                group_rate = pd.Series(0.265, index=out.index)  # IBM Telco baseline

            out["churn_probability"] = group_rate.clip(lower=0.001, upper=0.999)

            if "_tenure_q" in out.columns:
                out.drop(columns=["_tenure_q"], inplace=True)

        out["clv"] = calc.compute_clv_for_dataframe(out)
        logger.info(
            "CLV computed – mean=%.2f, median=%.2f, min=%.2f, max=%.2f",
            out["clv"].mean(),
            out["clv"].median(),
            out["clv"].min(),
            out["clv"].max(),
        )
        return out
