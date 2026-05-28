"""
Module tiền xử lý dữ liệu: làm sạch, transform, split.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from config.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """
    Pipeline tiền xử lý dữ liệu linh hoạt.

    Usage:
        preprocessor = Preprocessor()
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(df, target_col="label")
    """

    def __init__(self):
        self.scaler: Any = None
        self.label_encoder: Any = None
        self._fitted = False

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "mean",
        fill_value: Optional[Any] = None,
    ) -> pd.DataFrame:
        """
        Xử lý missing values.

        Args:
            df: DataFrame đầu vào.
            strategy: 'mean', 'median', 'mode', 'constant', 'drop'.
            fill_value: Giá trị fill nếu strategy='constant'.
        """
        df = df.copy()
        missing_cols = df.columns[df.isnull().any()].tolist()

        if not missing_cols:
            logger.info("Không có missing values.")
            return df

        logger.info(f"Xử lý missing ở các cột: {missing_cols}")

        if strategy == "drop":
            df = df.dropna()
        elif strategy == "constant" and fill_value is not None:
            df[missing_cols] = df[missing_cols].fillna(fill_value)
        else:
            for col in missing_cols:
                if strategy == "mean":
                    fill_val = df[col].mean()
                elif strategy == "median":
                    fill_val = df[col].median()
                elif strategy == "mode":
                    fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else 0
                else:
                    raise ValueError(f"Strategy '{strategy}' không hợp lệ.")
                df[col] = df[col].fillna(fill_val)

        logger.info(f"Còn {df.isnull().sum().sum()} missing values sau xử lý.")
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Loại bỏ dòng trùng lặp."""
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if before != after:
            logger.info(f"Đã loại bỏ {before - after} dòng trùng lặp.")
        return df

    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: list[str],
        method: str = "label",
    ) -> pd.DataFrame:
        """
        Mã hóa cột categorical.

        Args:
            df: DataFrame.
            columns: Danh sách cột cần encode.
            method: 'label' (LabelEncoder) hoặc 'onehot' (One-Hot).
        """
        df = df.copy()
        if method == "label":
            for col in columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                logger.info(f"Đã label-encode cột '{col}'.")
        elif method == "onehot":
            df = pd.get_dummies(df, columns=columns, drop_first=False)
            logger.info(f"Đã one-hot encode các cột: {columns}")
        else:
            raise ValueError(f"Method '{method}' không hợp lệ.")
        return df

    def scale_features(
        self,
        X: np.ndarray,
        method: str = "standard",
        fit: bool = True,
    ) -> np.ndarray:
        """
        Scale features.

        Args:
            X: Feature matrix.
            method: 'standard' (StandardScaler) hoặc 'minmax' (MinMaxScaler).
            fit: True nếu fit + transform, False nếu chỉ transform.
        """
        scaler_cls = StandardScaler if method == "standard" else MinMaxScaler
        if fit or not self._fitted:
            self.scaler = scaler_cls()
            X_scaled = self.scaler.fit_transform(X)
            self._fitted = True
        else:
            if self.scaler is None:
                raise RuntimeError("Scaler chưa được fit. Gọi fit=True trước.")
            X_scaled = self.scaler.transform(X)
        return X_scaled

    def train_val_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
        random_seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Chia dữ liệu thành train / validation / test.

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        test_size = test_size or config.data.train_test_split
        val_size = val_size or config.data.validation_split
        seed = random_seed or config.data.random_seed

        # train + val / test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        # train / val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=seed, stratify=y_temp
        )

        logger.info(
            f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def detect_column_types(self, df: pd.DataFrame) -> dict[str, list[str]]:
        """Phát hiện tự động kiểu dữ liệu các cột."""
        return {
            "numeric": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical": df.select_dtypes(include=["object", "category"]).columns.tolist(),
            "datetime": df.select_dtypes(include=["datetime64"]).columns.tolist(),
            "boolean": df.select_dtypes(include=["bool"]).columns.tolist(),
        }
