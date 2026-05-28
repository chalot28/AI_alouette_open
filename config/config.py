"""
Cấu hình trung tâm cho toàn bộ dự án.
Ưu tiên: environment variable > config file > default value.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from dotenv import load_dotenv


# ──────────────────────────────────────────────
# Load .env
# ──────────────────────────────────────────────
load_dotenv()


def _resolve_path(path: str) -> Path:
    """Resolve đường dẫn tương đối so với project root."""
    root = Path(__file__).resolve().parent.parent
    return root / path


# ──────────────────────────────────────────────
# Data Config
# ──────────────────────────────────────────────
@dataclass
class DataConfig:
    raw_dir: Path = _resolve_path(os.getenv("RAW_DATA_DIR", "data/raw"))
    processed_dir: Path = _resolve_path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
    interim_dir: Path = _resolve_path(os.getenv("INTERIM_DATA_DIR", "data/interim"))
    external_dir: Path = _resolve_path(os.getenv("EXTERNAL_DATA_DIR", "data/external"))

    train_test_split: float = float(os.getenv("TRAIN_TEST_SPLIT", "0.2"))
    validation_split: float = float(os.getenv("VALIDATION_SPLIT", "0.1"))
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))


# ──────────────────────────────────────────────
# Training Config
# ──────────────────────────────────────────────
@dataclass
class TrainingConfig:
    batch_size: int = int(os.getenv("BATCH_SIZE", "32"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "0.001"))
    num_epochs: int = int(os.getenv("NUM_EPOCHS", "100"))
    early_stopping_patience: int = int(os.getenv("EARLY_STOPPING_PATIENCE", "10"))
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────
# Model Config
# ──────────────────────────────────────────────
@dataclass
class ModelConfig:
    save_dir: Path = _resolve_path(os.getenv("MODEL_SAVE_DIR", "models"))
    input_dim: int = 784  # sẽ override theo dữ liệu thực tế
    hidden_dims: tuple[int, ...] = (256, 128)
    output_dim: int = 10
    dropout: float = 0.3
    activation: str = "relu"


# ──────────────────────────────────────────────
# Logging Config
# ──────────────────────────────────────────────
@dataclass
class LoggingConfig:
    level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: Path = _resolve_path(os.getenv("LOG_DIR", "logs"))
    use_rich: bool = True


# ──────────────────────────────────────────────
# Master Config
# ──────────────────────────────────────────────
@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    project_name: str = os.getenv("PROJECT_NAME", "alouette-ai")

    def __post_init__(self) -> None:
        # Tạo directories nếu chưa tồn tại
        dirs = [
            self.data.raw_dir,
            self.data.processed_dir,
            self.data.interim_dir,
            self.data.external_dir,
            self.model.save_dir,
            self.logging.log_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def set_seed(self, seed: Optional[int] = None) -> None:
        """Set seed cho reproducibility."""
        seed = seed or self.data.random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
# Singleton instance
# ──────────────────────────────────────────────
config = Config()
