"""
Training Loop — huấn luyện model với đầy đủ tính năng.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import Config
from src.models.base import BaseModel
from src.utils.logger import get_logger, get_rich_progress
from src.utils.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoardLogger,
)

logger = get_logger(__name__)


class Trainer:
    """
    Trainer tổng quát cho PyTorch models.

    Hỗ trợ:
      - Training / validation loop
      - Early stopping
      - Model checkpoint
      - TensorBoard logging
      - Gradient clipping
      - Learning rate scheduling
      - Mixed precision (AMP)
    """

    def __init__(
        self,
        model: BaseModel,
        config: Config,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        callbacks: Optional[list[Callback]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(
            device or config.training.device
        )
        self.model.to_device(self.device)

        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
        )
        self.scheduler = scheduler

        # Callbacks
        self.callbacks = callbacks or [
            EarlyStopping(patience=config.training.early_stopping_patience),
            ModelCheckpoint(
                save_dir=config.model.save_dir,
                monitor="val_loss",
                mode="min",
            ),
        ]

        # AMP (mixed precision)
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf")
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        """Train một epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = get_rich_progress(
            train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.num_epochs} [Train]",
        )

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                # Gradient clipping
                if hasattr(self.config, "grad_clip_norm"):
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            self.global_step += 1

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        # Step scheduler (nếu là one-cycle hoặc step per batch)
        if self.scheduler is not None and not isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            self.scheduler.step()

        return {"loss": avg_loss, "accuracy": accuracy}

    @torch.no_grad()
    def validate_epoch(self, val_loader: DataLoader) -> dict[str, float]:
        """Validate một epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = get_rich_progress(
            val_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.num_epochs} [Val]",
        )

        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return {"loss": avg_loss, "accuracy": accuracy}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
    ) -> dict[str, list[float]]:
        """
        Huấn luyện model.

        Args:
            train_loader: DataLoader cho training set.
            val_loader: DataLoader cho validation set (optional).
            epochs: Số epoch (override config).

        Returns:
            History dictionary.
        """
        num_epochs = epochs or self.config.training.num_epochs

        # Gọi callbacks: on_train_begin
        for cb in self.callbacks:
            cb.on_train_begin(self)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])

            # Validate
            val_metrics = train_metrics
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_acc"].append(val_metrics["accuracy"])

            # Step scheduler (plateau type)
            if self.scheduler is not None and isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step(val_metrics["loss"])

            # Log
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch + 1:3d}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.2f}% | "
                f"LR: {lr:.2e}"
            )

            # Gọi callbacks: on_epoch_end
            should_stop = False
            for cb in self.callbacks:
                cb.on_epoch_end(self, val_metrics["loss"])
                if hasattr(cb, "stopped") and cb.stopped:
                    should_stop = True

            if should_stop:
                logger.info(f"Early stopping triggered tại epoch {epoch + 1}.")
                break

        # Gọi callbacks: on_train_end
        for cb in self.callbacks:
            cb.on_train_end(self)

        return self.history
