"""
Training Callbacks — hook vào quá trình training.

Các callback:
  - Callback:        Base class
  - EarlyStopping:   Dừng sớm khi validation loss không cải thiện
  - ModelCheckpoint: Lưu model tốt nhất
  - TensorBoardLogger: Log lên TensorBoard
  - LRMonitor:       Theo dõi learning rate
  - ProgressBar:     Rich progress bar
"""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class Callback(ABC):
    """Abstract base cho tất cả callbacks."""

    @abstractmethod
    def on_train_begin(self, trainer: Any) -> None: ...

    @abstractmethod
    def on_epoch_end(self, trainer: Any, metric: float) -> None: ...

    @abstractmethod
    def on_train_end(self, trainer: Any) -> None: ...


# ──────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────
class EarlyStopping(Callback):
    """Dừng training sớm khi metric không cải thiện."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_metric = math.inf if mode == "min" else -math.inf
        self.counter = 0
        self.stopped = False

        self._check_improvement = (
            (lambda curr, best: curr < best - self.min_delta)
            if mode == "min"
            else (lambda curr, best: curr > best + self.min_delta)
        )

    def on_train_begin(self, trainer: Any) -> None:
        self.best_metric = math.inf if self.mode == "min" else -math.inf
        self.counter = 0
        self.stopped = False

    def on_epoch_end(self, trainer: Any, metric: float) -> None:
        if self._check_improvement(metric, self.best_metric):
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True

    def on_train_end(self, trainer: Any) -> None:
        pass


# ──────────────────────────────────────────────
# Model Checkpoint
# ──────────────────────────────────────────────
class ModelCheckpoint(Callback):
    """Lưu checkpoint mỗi khi metric cải thiện."""

    def __init__(
        self,
        save_dir: Path | str,
        monitor: str = "val_loss",
        mode: str = "min",
        filename: str = "best_model.pt",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.filename = filename
        self.best_metric = math.inf if mode == "min" else -math.inf

        self._check_improvement = (
            (lambda curr, best: curr < best)
            if mode == "min"
            else (lambda curr, best: curr > best)
        )

    def on_train_begin(self, trainer: Any) -> None:
        self.best_metric = math.inf if self.mode == "min" else -math.inf

    def on_epoch_end(self, trainer: Any, metric: float) -> None:
        if self._check_improvement(metric, self.best_metric):
            self.best_metric = metric
            save_path = self.save_dir / self.filename
            trainer.model.save_checkpoint(
                path=save_path,
                epoch=trainer.current_epoch,
                optimizer=trainer.optimizer,
                metrics={self.monitor: metric},
            )

    def on_train_end(self, trainer: Any) -> None:
        pass


# ──────────────────────────────────────────────
# TensorBoard Logger
# ──────────────────────────────────────────────
class TensorBoardLogger(Callback):
    """Log metrics lên TensorBoard."""

    def __init__(self, log_dir: Optional[Path | str] = None) -> None:
        self.log_dir = Path(log_dir) if log_dir else Path("logs/tensorboard")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer: Any = None

    def on_train_begin(self, trainer: Any) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:
            self.writer = None

    def on_epoch_end(self, trainer: Any, metric: float) -> None:
        if self.writer is None:
            return
        for key, values in trainer.history.items():
            if values:
                self.writer.add_scalar(key, values[-1], trainer.current_epoch)
        self.writer.add_scalar(
            "learning_rate",
            trainer.optimizer.param_groups[0]["lr"],
            trainer.current_epoch,
        )

    def on_train_end(self, trainer: Any) -> None:
        if self.writer is not None:
            self.writer.close()
