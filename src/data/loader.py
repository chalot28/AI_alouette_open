"""
Module load dữ liệu từ nhiều nguồn khác nhau.
Hỗ trợ: CSV, Parquet, JSON, Image folders, custom datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Load dữ liệu từ raw/ hoặc external/."""

    SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".json", ".jsonl", ".pkl", ".feather"}

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or config.data.raw_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load(self, filename: str, **kwargs: Any) -> pd.DataFrame:
        """
        Tự động phát hiện định dạng và load file.

        Args:
            filename: Tên file (có extension) trong data_dir.
            **kwargs: Truyền thêm cho pandas read_*.

        Returns:
            pd.DataFrame chứa dữ liệu.
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

        ext = filepath.suffix.lower()
        logger.info(f"Đang load dữ liệu từ: {filepath}")

        loaders = {
            ".csv": pd.read_csv,
            ".parquet": pd.read_parquet,
            ".json": pd.read_json,
            ".jsonl": pd.read_json,
            ".pkl": pd.read_pickle,
            ".feather": pd.read_feather,
        }

        loader = loaders.get(ext)
        if loader is None:
            raise ValueError(
                f"Định dạng '{ext}' không được hỗ trợ. "
                f"Hỗ trợ: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        extra_kwargs = {}
        if ext == ".csv":
            extra_kwargs = {"encoding": "utf-8", "low_memory": False}
        if ext == ".json":
            extra_kwargs = {"lines": True} if filename.endswith(".jsonl") else {}

        df = loader(filepath, **{**extra_kwargs, **kwargs})
        logger.info(f"Đã load {len(df)} rows, {len(df.columns)} columns.")
        return df

    def load_from_source(
        self,
        source: str,
        source_type: str = "csv",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load từ URL hoặc đường dẫn tuyệt đối.

        Args:
            source: URL hoặc đường dẫn.
            source_type: 'csv', 'json', 'parquet'.
            **kwargs: Truyền thêm.

        Returns:
            pd.DataFrame.
        """
        logger.info(f"Đang load từ nguồn: {source}")
        readers = {
            "csv": pd.read_csv,
            "json": pd.read_json,
            "parquet": pd.read_parquet,
        }
        reader = readers.get(source_type)
        if reader is None:
            raise ValueError(f"source_type '{source_type}' không hợp lệ.")
        return reader(source, **kwargs)

    def get_file_list(self, pattern: str = "*") -> list[Path]:
        """Liệt kê các file dữ liệu trong data_dir."""
        return sorted(self.data_dir.glob(pattern))
