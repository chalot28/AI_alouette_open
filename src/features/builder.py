"""
Feature Engineering — xây dựng feature từ dữ liệu thô.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureBuilder:
    """
    Xây dựng feature mới từ dữ liệu có sẵn.

    Hỗ trợ:
      - Tương tác (interaction)
      - Polynomial
      - Aggregation theo nhóm
      - Binning
      - Datetime features
      - Text features
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._feature_log: list[str] = []

    @property
    def feature_log(self) -> list[str]:
        return self._feature_log

    def add_interaction(
        self,
        col1: str,
        col2: str,
        operation: str = "multiply",
        name: Optional[str] = None,
    ) -> "FeatureBuilder":
        """Tạo interaction feature giữa 2 cột."""
        ops = {
            "multiply": lambda a, b: a * b,
            "divide": lambda a, b: a / (b + 1e-8),
            "add": lambda a, b: a + b,
            "subtract": lambda a, b: a - b,
        }
        if operation not in ops:
            raise ValueError(f"Operation '{operation}' không hợp lệ.")
        new_name = name or f"{col1}_{operation}_{col2}"
        self.df[new_name] = ops[operation](self.df[col1], self.df[col2])
        self._feature_log.append(f"Interaction: {new_name}")
        return self

    def add_polynomial(
        self,
        col: str,
        degree: int = 2,
        include_interaction: bool = True,
    ) -> "FeatureBuilder":
        """Thêm polynomial features (degree 2)."""
        from sklearn.preprocessing import PolynomialFeatures

        cols = [c for c in self.df.select_dtypes(include=[np.number]).columns if c != col]
        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=not include_interaction,
            include_bias=False,
        )
        poly_features = poly.fit_transform(self.df[[col] + cols])
        feature_names = poly.get_feature_names_out([col] + cols)
        new_df = pd.DataFrame(poly_features, columns=feature_names, index=self.df.index)
        # Merge, tránh trùng
        for c in new_df.columns:
            if c not in self.df.columns:
                self.df[c] = new_df[c]
                self._feature_log.append(f"Polynomial: {c}")
        return self

    def add_binned(
        self,
        col: str,
        bins: int = 5,
        labels: Optional[list[str]] = None,
        prefix: str = "bin",
    ) -> "FeatureBuilder":
        """Binning một numeric column."""
        new_name = f"{prefix}_{col}"
        self.df[new_name] = pd.cut(self.df[col], bins=bins, labels=labels)
        self._feature_log.append(f"Binned: {new_name} ({bins} bins)")
        return self

    def add_datetime_features(
        self,
        col: str,
        features: Optional[list[str]] = None,
    ) -> "FeatureBuilder":
        """Trích xuất features từ datetime column."""
        dt_col = pd.to_datetime(self.df[col])
        feature_map = {
            "year": dt_col.dt.year,
            "month": dt_col.dt.month,
            "day": dt_col.dt.day,
            "dayofweek": dt_col.dt.dayofweek,
            "hour": dt_col.dt.hour,
            "quarter": dt_col.dt.quarter,
            "is_weekend": (dt_col.dt.dayofweek >= 5).astype(int),
        }
        features = features or list(feature_map.keys())
        for f in features:
            if f in feature_map:
                new_name = f"{col}_{f}"
                self.df[new_name] = feature_map[f]
                self._feature_log.append(f"Datetime: {new_name}")
        return self

    def add_aggregate(
        self,
        group_col: str,
        agg_col: str,
        agg_func: str = "mean",
    ) -> "FeatureBuilder":
        """Tạo aggregate feature theo nhóm."""
        new_name = f"{group_col}_{agg_col}_{agg_func}"
        grouped = self.df.groupby(group_col)[agg_col].transform(agg_func)
        self.df[new_name] = grouped
        self._feature_log.append(f"Aggregate: {new_name}")
        return self

    def add_custom(self, func: Callable, name: str) -> "FeatureBuilder":
        """Thêm feature custom bằng hàm do người dùng định nghĩa."""
        self.df[name] = func(self.df)
        self._feature_log.append(f"Custom: {name}")
        return self

    def build(self) -> pd.DataFrame:
        """Trả về DataFrame với tất cả features đã tạo."""
        n_features = len(self._feature_log)
        logger.info(f"FeatureBuilder: đã tạo {n_features} features mới.")
        return self.df
