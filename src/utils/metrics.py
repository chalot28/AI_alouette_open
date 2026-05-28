"""
Metrics Tracker — theo dõi metrics trong quá trình training.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


class MetricsTracker:
    """
    Theo dõi và tính toán running average của metrics.

    Usage:
        tracker = MetricsTracker()
        tracker.update({"loss": 0.5, "accuracy": 85.0})
        tracker.update({"loss": 0.3, "accuracy": 90.0})
        avg = tracker.average()  # {"loss": 0.4, "accuracy": 87.5}
    """

    def __init__(self) -> None:
        self._metrics: dict[str, list[float]] = defaultdict(list)

    def reset(self) -> None:
        """Reset tất cả metrics."""
        self._metrics.clear()

    def update(self, metrics: dict[str, float]) -> None:
        """Cập nhật metrics từ batch."""
        for key, value in metrics.items():
            self._metrics[key].append(value)

    def average(self) -> dict[str, float]:
        """Tính average cho mỗi metric."""
        return {
            key: float(np.mean(values))
            for key, values in self._metrics.items()
        }

    def last(self) -> dict[str, float]:
        """Lấy giá trị mới nhất."""
        return {
            key: values[-1]
            for key, values in self._metrics.items()
        }

    def summary(self) -> str:
        """In summary các metrics."""
        avg = self.average()
        parts = [f"{k}: {v:.4f}" for k, v in avg.items()]
        return " | ".join(parts)
