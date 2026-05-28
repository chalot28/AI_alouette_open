"""
Inference — dự đoán với model đã trained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """
    Predictor cho inference.

    Usage:
        predictor = Predictor(model, device="cuda")
        predictions = predictor.predict(X_new)
    """

    def __init__(
        self,
        model: BaseModel,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device or next(model.parameters()).device)
        self.model.to_device(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor, DataLoader],
        batch_size: int = 32,
        return_proba: bool = False,
    ) -> np.ndarray:
        """
        Dự đoán nhãn / xác suất cho dữ liệu mới.

        Args:
            X: Dữ liệu đầu vào (numpy array, torch tensor, hoặc DataLoader).
            batch_size: Batch size nếu X là array/tensor.
            return_proba: True -> trả về probabilities, False -> trả về class.

        Returns:
            numpy array chứa kết quả dự đoán.
        """
        # Chuyển về DataLoader nếu cần
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
            loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size)
        elif isinstance(X, torch.Tensor):
            loader = DataLoader(TensorDataset(X.float()), batch_size=batch_size)
        elif isinstance(X, DataLoader):
            loader = X
        else:
            raise TypeError(f"Kiểu dữ liệu '{type(X)}' không được hỗ trợ.")

        all_outputs: list[np.ndarray] = []

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(self.device)
            else:
                inputs = batch.to(self.device)

            outputs = self.model(inputs)

            if return_proba:
                outputs = torch.softmax(outputs, dim=1)
            else:
                outputs = outputs.argmax(dim=1)

            all_outputs.append(outputs.cpu().numpy())

        return np.concatenate(all_outputs)

    @torch.no_grad()
    def predict_single(self, x: Union[np.ndarray, torch.Tensor]) -> int:
        """Dự đoán cho một sample duy nhất."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.unsqueeze(0).to(self.device)
        output = self.model(x)
        return output.argmax(dim=1).item()
