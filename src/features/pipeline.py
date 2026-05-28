"""
Pipeline xử lý features — kết hợp nhiều bước feature engineering.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Pipeline feature engineering tương thích scikit-learn.
    Cho phép kết hợp nhiều transformers.

    Usage:
        pipeline = FeaturePipeline(steps=[...])
        X_transformed = pipeline.fit_transform(X)
    """

    def __init__(self, steps: Optional[list[tuple[str, Any]]] = None):
        self.steps = steps or []

    def add_step(self, name: str, transformer: Any) -> "FeaturePipeline":
        """Thêm một bước vào pipeline."""
        self.steps.append((name, transformer))
        return self

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeaturePipeline":
        for name, step in self.steps:
            logger.info(f"Fitting step: {name}")
            if hasattr(step, "fit"):
                step.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        X_out = X.copy()
        for name, step in self.steps:
            logger.info(f"Transforming step: {name}")
            if hasattr(step, "transform"):
                X_out = step.transform(X_out)
            else:
                X_out = step(X_out)
        return X_out

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X, y)
