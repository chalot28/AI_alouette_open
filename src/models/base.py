"""
Base Model — Abstract class cho tất cả models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(nn.Module, ABC):
    """
    Abstract base cho tất cả models trong project.

    Cung cấp:
      - save / load checkpoint
      - count parameters
      - device management
      - forward abstract method
    """

    def __init__(self) -> None:
        super().__init__()
        self._device = torch.device("cpu")

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — phải implement ở subclass."""
        ...

    def to_device(self, device: torch.device | str) -> "BaseModel":
        """Đẩy model lên device cụ thể."""
        self._device = torch.device(device)
        return self.to(self._device)

    @property
    def device(self) -> torch.device:
        return self._device

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Đếm số lượng parameters."""
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def save_checkpoint(
        self,
        path: Path | str,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[dict[str, float]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Lưu checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "model_class": self.__class__.__name__,
            "epoch": epoch,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if metrics is not None:
            checkpoint["metrics"] = metrics
        if extra is not None:
            checkpoint.update(extra)

        torch.save(checkpoint, path)
        logger.info(f"Đã lưu checkpoint tại: {path}")

    def load_checkpoint(
        self,
        path: Path | str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[str] = None,
    ) -> dict[str, Any]:
        """Load checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")

        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Đã load checkpoint từ: {path} (epoch {checkpoint.get('epoch', '?')})")
        return checkpoint

    def summary(self) -> str:
        """In thông tin tóm tắt model."""
        n_params = self.count_parameters()
        n_total = self.count_parameters(trainable_only=False)
        lines = [
            f"{'='*60}",
            f"  Model:     {self.__class__.__name__}",
            f"  Device:    {self.device}",
            f"  Params:    {n_params:,} trainable / {n_total:,} total",
            f"{'='*60}",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────
# MLP Example — minh họa
# ──────────────────────────────────────────────
class MLP(BaseModel):
    """Multi-Layer Perceptron đơn giản."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] = (256, 128),
        output_dim: int = 10,
        dropout: float = 0.3,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                act_fn(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Khởi tạo weights theo Kaiming He."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
