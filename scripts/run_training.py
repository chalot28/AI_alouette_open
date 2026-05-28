"""
Script chính để chạy training pipeline.

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --epochs 50 --batch_size 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Thêm project root vào PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, TensorDataset

from config.config import config
from src.data.loader import DataLoader as DataLoader_
from src.data.preprocessor import Preprocessor
from src.models.base import MLP
from src.models.evaluate import Evaluator
from src.models.train import Trainer
from src.utils.helpers import Timer
from src.utils.logger import get_logger, console

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alouette AI — Training Script")
    parser.add_argument("--epochs", type=int, default=None, help="Số epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--data", type=str, default=None, help="Đường dẫn file dữ liệu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Override config từ CLI ──
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr

    # ── Seed ──
    config.set_seed()
    logger.info(f"Device: {config.training.device}")
    console.rule("[bold blue]Alouette AI — Training Pipeline[/bold blue]")

    # ── Load data ──
    with Timer("Data loading"):
        data_loader = DataLoader_()
        if args.data:
            df = data_loader.load_from_source(args.data)
        else:
            # Nếu không có data thật, tạo synthetic data demo
            logger.warning("Không tìm thấy file dữ liệu. Đang tạo dữ liệu synthetic...")
            import numpy as np
            np.random.seed(config.data.random_seed)

            n_samples = 5000
            X = np.random.randn(n_samples, 20).astype(np.float32)
            y = (X.sum(axis=1) > 0).astype(int)
            df = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
            df["label"] = y

    # ── Preprocess ──
    with Timer("Preprocessing"):
        preprocessor = Preprocessor()
        df = preprocessor.remove_duplicates(df)
        df = preprocessor.handle_missing_values(df)

        y = df["label"].values
        X = df.drop(columns=["label"]).values
        # Chuẩn hóa
        X = preprocessor.scale_features(X, method="standard")

        X_train, X_val, X_test, y_train, y_val, y_test = (
            preprocessor.train_val_test_split(X, y)
        )

    # ── DataLoaders ──
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long(),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size * 2,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size * 2,
        shuffle=False,
        num_workers=0,
    )

    # ── Model ──
    input_dim = X_train.shape[1]
    model = MLP(
        input_dim=input_dim,
        hidden_dims=(128, 64),
        output_dim=len(set(y)),
        dropout=0.3,
    )
    logger.info(f"\n{model.summary()}")

    # ── Criterion & Optimizer ──
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ── Trainer ──
    trainer = Trainer(
        model=model,
        config=config,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # ── Training ──
    with Timer("Training"):
        history = trainer.fit(train_loader, val_loader)

    # ── Evaluation ──
    evaluator = Evaluator(model)
    results = evaluator.evaluate(test_loader, num_classes=len(set(y)))

    # ── Save final model ──
    final_path = config.model.save_dir / "final_model.pt"
    model.save_checkpoint(
        path=final_path,
        epoch=config.training.num_epochs,
        optimizer=optimizer,
        metrics={"test_accuracy": results["accuracy"]},
    )

    logger.info("✅ Training hoàn tất!")
    logger.info(f"Best model: {config.model.save_dir / 'best_model.pt'}")
    logger.info(f"Final model: {final_path}")


if __name__ == "__main__":
    import numpy as np
    import pandas as pd  # noqa: F811
    main()
