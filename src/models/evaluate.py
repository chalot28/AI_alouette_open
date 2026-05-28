"""
Evaluation — đánh giá model trên test set.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """
    Đánh giá model với nhiều metrics.

    Usage:
        evaluator = Evaluator(model, device="cuda")
        results = evaluator.evaluate(test_loader)
    """

    def __init__(
        self,
        model: BaseModel,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device or next(model.parameters()).device)
        self.model.to_device(self.device)

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        num_classes: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Đánh giá model trên test_loader.

        Returns:
            Dict chứa: accuracy, precision, recall, f1, confusion_matrix, ...
        """
        self.model.eval()

        all_preds: list[int] = []
        all_probs: list[np.ndarray] = []
        all_targets: list[int] = []
        total_loss = 0.0

        criterion = torch.nn.CrossEntropyLoss()

        pbar = tqdm(test_loader, desc="Evaluating")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_probs = np.array(all_probs)

        # Metrics
        results: dict[str, Any] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "avg_loss": total_loss / len(test_loader),
            "classification_report": classification_report(
                y_true, y_pred, zero_division=0, output_dict=True
            ),
        }

        # ROC-AUC (nếu binary hoặc multi-class)
        if num_classes is not None and num_classes == 2:
            results["roc_auc"] = roc_auc_score(y_true, y_probs[:, 1])
        elif num_classes is not None and num_classes > 2:
            try:
                results["roc_auc_ovr"] = roc_auc_score(
                    y_true, y_probs, multi_class="ovr"
                )
            except ValueError:
                logger.warning("Không thể tính ROC-AUC.")

        self._log_results(results)
        return results

    @torch.no_grad()
    def predict_proba(
        self,
        loader: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict probabilities cho mỗi class."""
        self.model.eval()
        all_probs: list[np.ndarray] = []
        all_targets: list[int] = []

        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.numpy())

        return np.array(all_probs), np.array(all_targets)

    def _log_results(self, results: dict[str, Any]) -> None:
        """In kết quả đánh giá."""
        logger.info("=" * 60)
        logger.info("KẾT QUẢ ĐÁNH GIÁ")
        logger.info("=" * 60)
        logger.info(f"Accuracy:          {results['accuracy']:.4f}")
        logger.info(f"Precision (macro): {results['precision_macro']:.4f}")
        logger.info(f"Recall (macro):    {results['recall_macro']:.4f}")
        logger.info(f"F1 Score (macro):  {results['f1_macro']:.4f}")
        logger.info(f"F1 Score (weight): {results['f1_weighted']:.4f}")
        logger.info(f"Avg Loss:          {results['avg_loss']:.4f}")
        if "roc_auc" in results:
            logger.info(f"ROC-AUC:           {results['roc_auc']:.4f}")
        logger.info("=" * 60)
