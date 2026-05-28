"""
Data Augmentation — tăng cường dữ liệu cho cả tabular, image, text.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataAugmenter:
    """Tăng cường dữ liệu với các kỹ thuật khác nhau."""

    @staticmethod
    def add_gaussian_noise(
        df: pd.DataFrame,
        columns: Optional[list[str]] = None,
        mean: float = 0.0,
        std: float = 0.01,
    ) -> pd.DataFrame:
        """Thêm Gaussian noise vào numeric columns."""
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        result = df.copy()
        noise = np.random.normal(mean, std, size=(len(df), len(cols)))
        result[cols] = result[cols] + noise
        logger.info(f"Đã thêm Gaussian noise (std={std}) vào {len(cols)} cột.")
        return result

    @staticmethod
    def smote_synthetic(
        X: np.ndarray,
        y: np.ndarray,
        target_class: Any,
        n_synthetic: int,
        k_neighbors: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        SMOTE đơn giản: sinh synthetic samples cho class thiểu số.
        (Phiên bản minh họa — dùng imbalanced-learn trong production.)
        """
        from sklearn.neighbors import NearestNeighbors

        # Lọc samples của class mục tiêu
        X_target = X[y == target_class]
        n_samples = X_target.shape[0]

        if n_samples < k_neighbors + 1:
            raise ValueError(
                f"Class '{target_class}' chỉ có {n_samples} samples, "
                f"cần ít nhất {k_neighbors + 1}."
            )

        # Tìm k-nearest neighbors
        nn = NearestNeighbors(n_neighbors=k_neighbors)
        nn.fit(X_target)
        _, indices = nn.kneighbors(X_target)

        synthetic_samples = []
        for _ in range(n_synthetic):
            idx = np.random.randint(0, n_samples)
            neighbor_idx = np.random.choice(indices[idx])
            diff = X_target[neighbor_idx] - X_target[idx]
            gap = np.random.random()
            synthetic = X_target[idx] + gap * diff
            synthetic_samples.append(synthetic)

        X_new = np.vstack([X, np.array(synthetic_samples)])
        y_new = np.hstack([y, np.full(n_synthetic, target_class)])
        return X_new, y_new
