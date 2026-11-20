"""
Churn Classifier Module
========================
LightGBM-based binary classifier for customer churn prediction.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for cross-package imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import logging
from typing import Dict, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from config.settings import LIGHTGBM_PARAMS, MODELS_DIR
from src.data_pipeline.feature_engineering import prepare_model_data

logger = logging.getLogger(__name__)

# Default model save path
_DEFAULT_MODEL_PATH = MODELS_DIR / "churn_lgbm_model.pkl"


class ChurnClassifier:
    """LightGBM churn classifier with training, evaluation, and persistence."""

    def __init__(self, params: Optional[Dict] = None) -> None:
        """
        Initialise the classifier.

        Parameters
        ----------
        params : dict, optional
            LightGBM hyper-parameters.  Falls back to ``LIGHTGBM_PARAMS``
            defined in *config/settings.py* when not provided.
        """
        self.params: Dict = params if params is not None else LIGHTGBM_PARAMS.copy()
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names: Optional[list] = None

    # ------------------------------------------------------------------ train
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, float]:
        """
        Train a LightGBM model with early stopping.

        Parameters
        ----------
        X_train, y_train : training data and labels.
        X_val, y_val : validation data and labels used for early stopping.

        Returns
        -------
        dict
            Evaluation metrics on the **validation** set:
            accuracy, precision, recall, f1, auc_roc, auc_pr.
        """
        self.feature_names = list(X_train.columns)

        # Separate early_stopping_rounds from the constructor params
        fit_params: Dict = {}
        model_params = self.params.copy()

        early_stopping = model_params.pop("early_stopping_rounds", 30)
        fit_params["callbacks"] = [
            lgb.early_stopping(stopping_rounds=early_stopping, verbose=False),
            lgb.log_evaluation(period=0),  # suppress iteration logs
        ]

        self.model = lgb.LGBMClassifier(**model_params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            **fit_params,
        )

        # Evaluate on validation set
        metrics = self._evaluate(X_val, y_val)
        logger.info("Validation metrics: %s", metrics)
        return metrics

    # -------------------------------------------------------------- predict
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return churn **probabilities** for the positive class."""
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded yet.")
        return self.model.predict_proba(X)[:, 1]

    def predict_class(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary churn predictions using *threshold*."""
        probas = self.predict(X)
        return (probas >= threshold).astype(int)

    # ------------------------------------------------- feature importance
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return a DataFrame with columns ``feature``, ``importance``,
        ``importance_type`` containing both *gain* and *split* importances.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded yet.")

        rows = []
        for imp_type in ("gain", "split"):
            importances = self.model.booster_.feature_importance(importance_type=imp_type)
            for name, value in zip(self.feature_names, importances):
                rows.append(
                    {"feature": name, "importance": value, "importance_type": imp_type}
                )

        df = pd.DataFrame(rows)
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    # --------------------------------------------------------- persistence
    def save_model(self, path: Optional[str] = None) -> None:
        """Save the trained model to disk via *joblib*."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        save_path = Path(path) if path else _DEFAULT_MODEL_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_names": self.feature_names, "params": self.params},
            save_path,
        )
        logger.info("Model saved to %s", save_path)

    def load_model(self, path: Optional[str] = None) -> None:
        """Load a previously saved model from disk."""
        load_path = Path(path) if path else _DEFAULT_MODEL_PATH
        data = joblib.load(load_path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.params = data.get("params", self.params)
        logger.info("Model loaded from %s", load_path)

    # ------------------------------------------------ end-to-end helper
    @staticmethod
    def train_and_evaluate(
        df: pd.DataFrame,
    ) -> Tuple["ChurnClassifier", Dict[str, float], pd.DataFrame, pd.Series]:
        """
        End-to-end convenience method.

        1. Calls ``prepare_model_data`` to get *(X, y)*.
        2. Performs an 80 / 20 stratified train / test split.
        3. Trains the classifier with a 75 / 25 inner split for early stopping.
        4. Evaluates on the held-out test set.

        Returns
        -------
        tuple
            (classifier, metrics_dict, X_test, y_test)
        """
        X, y = prepare_model_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Further split training data for early-stopping validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
        )

        classifier = ChurnClassifier()
        classifier.train(X_tr, y_tr, X_val, y_val)

        # Final evaluation on the true hold-out set
        metrics = classifier._evaluate(X_test, y_test)
        logger.info("Test metrics: %s", metrics)

        return classifier, metrics, X_test, y_test

    # --------------------------------------------------------- internal
    def _evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Compute standard classification metrics."""
        probas = self.predict(X)
        preds = (probas >= 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "auc_roc": float(roc_auc_score(y, probas)),
            "auc_pr": float(average_precision_score(y, probas)),
        }
