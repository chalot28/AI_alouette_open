"""
LogAnalyzer Training - NHẸ & NHANH
Model ~5M params tối ưu cho CPU.
"""

import logging
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)
console = __import__("rich.console", fromlist=["Console"]).Console()

# ── Config ──
ERROR_CSV = "D:/alouette-AI/data/raw/error_logs_500k.csv"
NORMAL_CSV = "D:/alouette-AI/data/raw/normal_logs_500k.csv"
MAX_SEQ = 256  # log ngắn ~100-200 ký tự, 256 dư dả
BATCH_SIZE = 256  # tăng batch, tận dụng CPU vectorization
LR = 3e-4
SUBSIZE = 100_000
EPOCHS = 3  # binary classification, hội tụ nhanh
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Import tokenizer từ LogAnalyzer hiện tại ──
from src.models.log_analyzer import PAD_IDX, VOCAB_SIZE, tokenize_log

log.info(
    f"Device: {DEVICE} | MaxSeq: {MAX_SEQ} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}"
)

# ═══════════════════════════════════════════
# 1. DATASET
# ═══════════════════════════════════════════


class FastLogDataset(Dataset):
    """Pre-tokenize ngay khi init — đọc từ tensor siêu nhanh."""

    def __init__(self, error_csv, normal_csv, max_seq, subsize=None):
        self.max_seq = max_seq

        df_e = pd.read_csv(error_csv, header=None, nrows=subsize)
        df_n = pd.read_csv(normal_csv, header=None, nrows=subsize)

        err_text = (
            df_e.iloc[:, 0].astype(str)
            + ","
            + df_e.iloc[:, 1].astype(str)
            + ","
            + df_e.iloc[:, 2].astype(str)
            + ","
            + df_e.iloc[:, 3].astype(str)
            + ","
            + df_e.iloc[:, 4].astype(str)
        )
        norm_text = (
            df_n.iloc[:, 0].astype(str)
            + ","
            + df_n.iloc[:, 1].astype(str)
            + ","
            + df_n.iloc[:, 2].astype(str)
            + ","
            + df_n.iloc[:, 3].astype(str)
        )

        texts = err_text.tolist() + norm_text.tolist()
        labels = [1] * len(err_text) + [0] * len(norm_text)

        # Pre-tokenize: numpy vectorized
        log.info("Tokenizing dataset...")
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.:/\\[](){}!@#$%^&*+=,;?<>'\"`~| \t\n\r"
        char_map = {ch: i + 1 for i, ch in enumerate(chars)}
        n = len(texts)
        arr = np.zeros((n, max_seq), dtype=np.int64)
        for i, t in enumerate(texts):
            for j, ch in enumerate(t[:max_seq]):
                arr[i, j] = char_map.get(ch, 0)
            if (i + 1) % 50000 == 0:
                log.info(f"  {i + 1}/{n}")
        self.tokens = torch.from_numpy(arr).long()
        self.labels = torch.tensor(labels, dtype=torch.long)
        log.info(f"Dataset: {n:,} ({len(err_text):,} err + {len(norm_text):,} normal)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]


def collate_fn(batch):
    tokens = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return tokens, labels


# ═══════════════════════════════════════════
# 2. LIGHTWEIGHT MODEL (~5M params)
# ═══════════════════════════════════════════


class LogAnalyzerLight(nn.Module):
    """
    LogAnalyzer bản nhẹ: 512 tokens → CNN×3 (512→64) → Transformer×4 → Classifier.
    ~5M parameters.
    """

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        d_model=256,
        n_layers=2,
        n_heads=8,
        d_ff=512,
        max_seq=256,
        num_classes=2,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq = max_seq

        # Embedding
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.drop = nn.Dropout(0.1)

        # Position encoding (sinusoidal, không học)
        pe = torch.zeros(max_seq, d_model)
        pos = torch.arange(0, max_seq, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

        # Downsample: AdaptiveAvgPool 256→32 (1 phát, zero params, siêu nhanh)
        self.downsample = nn.AdaptiveAvgPool1d(32)
        self.cnn_out = 32

        # Transformer blocks (có residual, pre-norm)
        from src.models.log_analyzer import TransformerBlock

        self.tf_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_kv_heads=4,
                    d_ff=d_ff,
                    dropout=0.1,
                )
                for _ in range(n_layers)
            ]
        )
        self.tf_norm = nn.LayerNorm(d_model)

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(self, x):
        B, S = x.shape
        x = self.embed(x) * math.sqrt(self.d_model)
        x = x + self.pe[:S].unsqueeze(0)
        x = self.drop(x)

        # Downsample: (B, d, 256) → (B, d, 32)
        x = x.transpose(1, 2)
        x = self.downsample(x)
        x = x.transpose(1, 2)

        # Transformer
        for block in self.tf_blocks:
            x = block(x)
        x = self.tf_norm(x)

        # Global pooling
        x = x.mean(dim=1)

        # Classifier
        return self.classifier(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════
# 3. TRAINING
# ═══════════════════════════════════════════


def train():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(os.cpu_count() or 8)
    torch.set_num_interop_threads(2)

    save_dir = Path("models/log_analyzer")
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "checkpoint.pt"

    # Dataset
    dataset = FastLogDataset(ERROR_CSV, NORMAL_CSV, MAX_SEQ, SUBSIZE)

    # Split
    n = len(dataset)
    train_n = int(0.8 * n)
    val_n = int(0.1 * n)
    test_n = n - train_n - val_n
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [train_n, val_n, test_n],
        generator=torch.Generator().manual_seed(42),
    )
    log.info(f"Split: train={train_n:,} val={val_n:,} test={test_n:,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Model
    model = LogAnalyzerLight(max_seq=MAX_SEQ)
    model.to(DEVICE)
    params = model.count_params()
    log.info(f"Model parameters: {params:,}")
    log.info(f"Context length: {MAX_SEQ}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_epoch = 0
    best_val_loss = float("inf")
    patience = 3
    wait = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Resume từ checkpoint nếu có
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        history = ckpt["history"]
        best_val_loss = ckpt["best_val_loss"]
        log.info(f"Resumed from checkpoint: epoch {start_epoch}/{EPOCHS}")

    console.rule("[bold green]BẮT ĐẦU TRAINING[/]")
    print(f"  • Model: LogAnalyzerLight ({params:,} params)")
    print(f"  • Data:  {n:,} samples (100K err + 100K normal)")
    print(f"  • MaxSeq: {MAX_SEQ} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS} | LR: {LR}")
    print(f"  • Device: {DEVICE}")
    if start_epoch > 0:
        print(f"  • Resumed at epoch {start_epoch}/{EPOCHS}")
    print()

    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        n_batches = len(train_loader)
        print(f"  Epoch {epoch + 1}/{EPOCHS} - {n_batches} batches")
        for batch_idx, (tokens, labels) in enumerate(train_loader):
            # 10 batch đầu in ngay để biết đang chạy (JIT warmup lâu)
            early_log = batch_idx < 10
            tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(tokens)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            _, preds = outputs.max(1)
            train_total += labels.size(0)
            train_correct += preds.eq(labels).sum().item()

            if (
                early_log
                or (batch_idx + 1) % (n_batches // 50) == 0
                or batch_idx == n_batches - 1
            ):
                pct = (batch_idx + 1) / n_batches * 100
                avg_loss = train_loss / (batch_idx + 1)
                avg_acc = 100 * train_correct / max(train_total, 1)
                print(
                    f"  [{pct:4.0f}%] loss={avg_loss:.4f} acc={avg_acc:.2f}%", end="\r"
                )
        print()
        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # ── Val ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for tokens, labels in val_loader:
                tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
                outputs = model(tokens)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        log.info(
            f"Epoch {epoch + 1:2d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"LR: {lr_now:.2e} | {elapsed:.0f}s"
        )

        # ── Save checkpoint mỗi epoch ──
        save_dir = Path("models/log_analyzer")
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "config": {"max_seq": MAX_SEQ, "num_classes": 2, "params": params},
            "best_val_loss": best_val_loss,
        }
        torch.save(checkpoint, save_dir / "checkpoint.pt")

        # Save best model riêng
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            log.info(f"  ✓ Best model updated (val_loss={val_loss:.4f})")
        else:
            wait += 1
            if wait >= patience:
                log.info(f"Early stop at epoch {epoch + 1}")
                break

    # ── Save final ──
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"max_seq": MAX_SEQ, "num_classes": 2, "params": params},
            "history": history,
        },
        save_dir / "final_model.pt",
    )
    log.info(f"Final model saved to {save_dir / 'final_model.pt'}")

    # ── Kết quả ──
    print("\n" + "=" * 50)
    print("KẾT QUẢ TRAINING")
    print("=" * 50)
    print(f"  Model: LogAnalyzerLight ({params:,} params)")
    print(f"  Dataset: {n:,} samples ({SUBSIZE:,} err + {SUBSIZE:,} normal)")
    print(f"  Context: {MAX_SEQ} tokens")
    print(
        f"  Batch: {BATCH_SIZE} | LR: {LR} | Epochs done: {len(history['train_loss'])}"
    )
    print(f"  Device: {DEVICE}")
    print()
    best_idx = int(np.argmin(history["val_loss"]))
    print(
        f"  Best Val Loss:  {history['val_loss'][best_idx]:.4f} (epoch {best_idx + 1})"
    )
    print(f"  Best Val Acc:   {history['val_acc'][best_idx]:.2f}%")
    print(f"  Final Train Acc: {history['train_acc'][-1]:.2f}%")
    print(f"  Final Val Acc:   {history['val_acc'][-1]:.2f}%")
    print("=" * 50)

    return model, history


if __name__ == "__main__":
    train()
