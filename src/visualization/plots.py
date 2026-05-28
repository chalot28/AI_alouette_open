"""
Visualization — các hàm vẽ biểu đồ phục vụ EDA và report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Style
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (12, 8),
    "figure.dpi": 120,
    "font.size": 12,
})


class Plotter:
    """
    Vẽ các biểu đồ phục vụ EDA, training monitoring, evaluation.

    Usage:
        plotter = Plotter(save_dir="reports/figures")
        plotter.plot_confusion_matrix(cm, class_names=[...])
    """

    def __init__(self, save_dir: Optional[Path | str] = None):
        self.save_dir = Path(save_dir) if save_dir else Path("reports/figures")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: list[str],
        title: str = "Confusion Matrix",
        normalize: bool = False,
        save: bool = True,
    ) -> plt.Figure:
        """Vẽ confusion matrix."""
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
        else:
            fmt = "d"

        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", values_format=fmt, colorbar=True)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save:
            self._save_fig(fig, title.lower().replace(" ", "_"))
        return fig

    def plot_training_history(
        self,
        history: dict[str, list[float]],
        save: bool = True,
    ) -> plt.Figure:
        """Vẽ biểu đồ loss và accuracy trong quá trình training."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(next(iter(history.values()))) + 1)

        # Loss
        ax = axes[0]
        if "train_loss" in history:
            ax.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
        if "val_loss" in history:
            ax.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss over Epochs", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy
        ax = axes[1]
        if "train_acc" in history:
            ax.plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
        if "val_acc" in history:
            ax.plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy over Epochs", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            self._save_fig(fig, "training_history")
        return fig

    def plot_feature_importance(
        self,
        importances: np.ndarray,
        feature_names: list[str],
        title: str = "Feature Importance",
        top_n: int = 20,
        save: bool = True,
    ) -> plt.Figure:
        """Vẽ biểu đồ feature importance (top N)."""
        indices = np.argsort(importances)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
        ax.barh(
            range(top_n),
            importances[indices][::-1],
            align="center",
            color="steelblue",
        )
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
        ax.set_xlabel("Importance")
        ax.set_title(title, fontweight="bold")
        plt.tight_layout()

        if save:
            self._save_fig(fig, "feature_importance")
        return fig

    def plot_class_distribution(
        self,
        y: pd.Series | np.ndarray,
        title: str = "Class Distribution",
        save: bool = True,
    ) -> plt.Figure:
        """Vẽ phân bố class."""
        fig, ax = plt.subplots()
        if isinstance(y, pd.Series):
            y.value_counts().plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
        else:
            unique, counts = np.unique(y, return_counts=True)
            ax.bar(unique, counts, color="steelblue", edgecolor="black")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title(title, fontweight="bold")
        plt.tight_layout()

        if save:
            self._save_fig(fig, "class_distribution")
        return fig

    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        title: str = "Correlation Heatmap",
        figsize: tuple[int, int] = (14, 10),
        save: bool = True,
    ) -> plt.Figure:
        """Vẽ heatmap tương quan giữa các features."""
        fig, ax = plt.subplots(figsize=figsize)
        corr = df.select_dtypes(include=[np.number]).corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=False,
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontweight="bold")
        plt.tight_layout()

        if save:
            self._save_fig(fig, "correlation_heatmap")
        return fig

    def _save_fig(self, fig: plt.Figure, name: str) -> None:
        """Lưu figure ra save_dir."""
        path = self.save_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Đã lưu figure: {path}")
