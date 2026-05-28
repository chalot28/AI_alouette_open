"""
Helper utilities — các hàm tiện ích dùng chung.
"""

from __future__ import annotations

import json
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable


def ensure_dir(path: Path | str) -> Path:
    """Tạo directory nếu chưa tồn tại."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, filepath: Path | str, indent: int = 2) -> None:
    """Save dict/list ra file JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(filepath: Path | str) -> Any:
    """Load dữ liệu từ file JSON."""
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def count_parameters(model: Any, trainable_only: bool = True) -> int:
    """Đếm số parameters của model."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


class Timer:
    """
    Context manager để đo thời gian thực thi.

    Usage:
        with Timer("Training") as t:
            model.train(...)
        print(t.elapsed)  # seconds
    """

    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self.start_time

    def __str__(self) -> str:
        if self.elapsed < 60:
            return f"{self.name}: {self.elapsed:.2f}s"
        return f"{self.name}: {self.elapsed / 60:.2f}m"
