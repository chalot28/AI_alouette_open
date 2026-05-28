"""
Huấn luyện LogAnalyzer với 1M log samples (500K error + 500K normal).
Model ~5M parameters, character-level Transformer.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from config.config import Config, DataConfig, LoggingConfig, ModelConfig, TrainingConfig
from src.models.log_analyzer import LogAnalyzer, tokenize_log
from src.models.train import Trainer
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────


class LogDataset(Dataset):
    """Dataset cho log classification: error vs not-error."""

    def __init__(self, error_csv: str, normal_csv: str, max_seq_len: int = 2048):
        self.max_seq_len = max_seq_len

        # Load error logs → label=1
        logger.info(f"Loading error logs from {error_csv}")
        df_e = pd.read_csv(error_csv, header=None)
        df_e.columns = ["level", "ts", "service", "exception", "message"]
        errors = df_e.values.tolist()

        # Load normal logs → label=0
        logger.info(f"Loading normal logs from {normal_csv}")
        df_n = pd.read_csv(normal_csv, header=None)
        df_n.columns = ["level", "ts", "service", "message"]
        normals = df_n.values.tolist()

        # Build (text, label) pairs
        self.samples = []
        for row in errors:
            text = f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}"
            self.samples.append((text, 1))
        for row in normals:
            text = f"{row[0]},{row[1]},{row[2]},{row[3]}"
            self.samples.append((text, 0))

        logger.info(
            f"Dataset: {len(self.samples):,} samples ({len(errors):,} error / {len(normals):,} normal)"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        tokens = tokenize_log(text, self.max_seq_len)
        return tokens, torch.tensor(label, dtype=torch.long)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────


def main():
    ERROR_CSV = "D:/alouette-AI/data/raw/error_logs_500k.csv"
    NORMAL_CSV = "D:/alouette-AI/data/raw/normal_logs_500k.csv"
    MAX_SEQ_LEN = 2048
    BATCH_SIZE = 64
    NUM_EPOCHS = 20
    LR = 3e-4

    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%m/%d/%y %H:%M:%S",
    )

    # Config
    cfg = Config(
        training=TrainingConfig(
            batch_size=BATCH_SIZE,
            learning_rate=LR,
            num_epochs=NUM_EPOCHS,
            early_stopping_patience=5,
            device="cuda" if torch.cuda.is_available() else "cpu",
        ),
        model=ModelConfig(save_dir=Path("models/log_analyzer")),
    )
    cfg.set_seed(42)
    logger.info(f"Device: {cfg.training.device}")

    # Dataset
    dataset = LogDataset(ERROR_CSV, NORMAL_CSV, max_seq_len=MAX_SEQ_LEN)

    # Split: 80% train, 10% val, 10% test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info(f"Split: train={train_size:,} val={val_size:,} test={test_size:,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Model
    model = LogAnalyzer(max_seq_len=MAX_SEQ_LEN, num_classes=2)
    logger.info(f"\n{model.summary()}")
    total_params = model.count_parameters()
    logger.info(f"Total parameters: {total_params:,}")

    # Optimizer & criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # Trainer
    trainer = Trainer(
        model=model,
        config=cfg,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # Train
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU HUẤN LUYỆN")
    logger.info("=" * 60)
    history = trainer.fit(train_loader, val_loader, epochs=NUM_EPOCHS)

    # Save final model
    final_path = Path("models/log_analyzer/final_model.pt")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "max_seq_len": MAX_SEQ_LEN,
                "num_classes": 2,
                "total_params": total_params,
            },
            "history": history,
        },
        final_path,
    )
    logger.info(f"Final model saved to {final_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("KẾT QUẢ HUẤN LUYỆN")
    print("=" * 60)
    print(f"  Model: LogAnalyzer ({total_params:,} params)")
    print(f"  Dataset: {len(dataset):,} samples (error + normal)")
    print(f"  Batch size: {BATCH_SIZE} | LR: {LR} | Epochs: {NUM_EPOCHS}")
    print(f"  Device: {cfg.training.device}")
    print()
    if history.get("val_loss"):
        best_idx = int(np.argmin(history["val_loss"]))
        print(
            f"  Best Val Loss:  {history['val_loss'][best_idx]:.4f} (epoch {best_idx + 1})"
        )
        print(f"  Best Val Acc:   {history['val_acc'][best_idx]:.2f}%")
    print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final Train Acc:  {history['train_acc'][-1]:.2f}%")
    if history.get("val_loss"):
        print(f"  Final Val Loss:   {history['val_loss'][-1]:.4f}")
        print(f"  Final Val Acc:    {history['val_acc'][-1]:.2f}%")
    print("=" * 60)
    print(f"Model saved: {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
