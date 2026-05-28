"""
LogAnalyzer PRO — Full quality ~6.8M params, 1M samples.
Tối ưu cho GPU training + AMP mixed precision.
"""

import argparse
import json
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Resolve paths relative to project root ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

# ── Config ──
parser = argparse.ArgumentParser()
parser.add_argument(
    "--error_csv", default=str(PROJECT_ROOT / "data/raw/error_logs_500k.csv")
)
parser.add_argument(
    "--normal_csv", default=str(PROJECT_ROOT / "data/raw/normal_logs_500k.csv")
)
parser.add_argument("--save_dir", default=str(PROJECT_ROOT / "models/log_analyzer_pro"))
parser.add_argument("--max_seq", type=int, default=1024)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

SAVE_DIR = Path(args.save_dir)
DEVICE = torch.device(args.device)
USE_AMP = DEVICE.type == "cuda"
log.info(
    f"Config: device={DEVICE} amp={USE_AMP} batch={args.batch_size} epochs={args.epochs}"
)

# ── Char tokenizer ──
LOG_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.:/\\[](){}!@#$%^&*+=,;?<>'\"`~| \t\n\r"
CHAR2IDX = {ch: i + 1 for i, ch in enumerate(LOG_CHARS)}
PAD_IDX = 0
VOCAB_SIZE = len(CHAR2IDX) + 1


def tokenize(texts, max_len):
    n = len(texts)
    arr = np.zeros((n, max_len), dtype=np.int32)
    for i, t in enumerate(texts):
        for j, ch in enumerate(t[:max_len]):
            arr[i, j] = CHAR2IDX.get(ch, 0)
    return torch.from_numpy(arr)


# ═══════════════════════════════════════════════
# FULL MODEL — 6.8M params
# ═══════════════════════════════════════════════


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_kv_heads=4, dropout=0.1):
        super().__init__()
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.n_groups = n_heads // n_kv_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, -1, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, -1, self.d_head).transpose(1, 2)
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head**0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        return self.out_proj(
            torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
        )


class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_kv_heads=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_up = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        r = x
        x = self.norm1(x)
        x = self.attn(x, mask)
        x = self.dropout(x)
        x = r + x
        r = x
        x = self.norm2(x)
        g = F.silu(self.ffn_gate(x))
        u = self.ffn_up(x)
        x = g * u
        x = self.ffn_down(x)
        x = self.dropout(x)
        return r + x


class Pooler(nn.Module):
    """AvgPool 1D: nén sequence xuống 128 — 0 params, siêu nhanh"""

    def __init__(self, d_model=256, pool_len=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(pool_len)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.transpose(1, 2)
        return self.norm(x)


class LogAnalyzerPro(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        d_model=256,
        n_layers=6,
        n_heads=8,
        n_kv_heads=4,
        d_ff=1024,
        max_seq=2048,
        num_classes=2,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        pe = torch.zeros(max_seq, d_model)
        pos = torch.arange(0, max_seq, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.drop = nn.Dropout(0.1)
        self.pooler = Pooler(d_model, pool_len=128)
        self.tf_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, n_kv_heads, d_ff, 0.1)
                for _ in range(n_layers)
            ]
        )
        self.tf_norm = nn.LayerNorm(d_model)
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
                nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, x):
        B, S = x.shape
        x = self.embed(x) * math.sqrt(self.d_model)
        x = x + self.pe[:S].unsqueeze(0)
        x = self.drop(x)
        x = self.pooler(x)
        for b in self.tf_blocks:
            x = b(x)
        x = self.tf_norm(x)
        return self.classifier(x.mean(dim=1))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════


class LogDataset(torch.utils.data.Dataset):
    def __init__(self, error_csv, normal_csv, max_seq):
        log.info(f"Loading {error_csv} & {normal_csv}")
        df_e = pd.read_csv(error_csv, header=None, nrows=None)
        df_n = pd.read_csv(normal_csv, header=None, nrows=None)
        err_t = (
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
        norm_t = (
            df_n.iloc[:, 0].astype(str)
            + ","
            + df_n.iloc[:, 1].astype(str)
            + ","
            + df_n.iloc[:, 2].astype(str)
            + ","
            + df_n.iloc[:, 3].astype(str)
        )
        texts = err_t.tolist() + norm_t.tolist()
        labels = [1] * len(err_t) + [0] * len(norm_t)
        log.info(f"Tokenizing {len(texts):,} samples...")
        self.tokens = tokenize(texts, max_seq)
        self.labels = torch.tensor(labels, dtype=torch.long)
        log.info(f"Dataset ready: {len(labels):,} | token shape={self.tokens.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]


# ═══════════════════════════════════════════════
# TRAINING (GPU + AMP)
# ═══════════════════════════════════════════════


def train():
    torch.manual_seed(42)
    np.random.seed(42)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    dataset = LogDataset(args.error_csv, args.normal_csv, args.max_seq)
    n = len(dataset)
    train_n, val_n = int(0.9 * n), int(0.05 * n)
    train_ds, val_ds, _ = torch.utils.data.random_split(
        dataset,
        [train_n, val_n, n - train_n - val_n],
        generator=torch.Generator().manual_seed(42),
    )

    nw = min(2, os.cpu_count() or 2) if DEVICE.type == "cuda" else 0
    train_loader = torch.utils.data.DataLoader(
        train_ds, args.batch_size, shuffle=True, num_workers=nw, pin_memory=USE_AMP
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, args.batch_size * 2, shuffle=False, num_workers=nw, pin_memory=USE_AMP
    )

    model = LogAnalyzerPro(max_seq=args.max_seq)
    model.to(DEVICE)
    # Bỏ torch.compile — compile lâu hơn lợi ích với model 6.8M trên Colab

    params = model.count_params()
    log.info(f"Model: {params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler() if USE_AMP else None

    ckpt_path = SAVE_DIR / "checkpoint.pt"
    start_epoch = 0
    best_val_loss = float("inf")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        history = ckpt["history"]
        best_val_loss = ckpt["best_val_loss"]
        log.info(f"Resumed: epoch {start_epoch}/{args.epochs}")

    print(f"\n{'=' * 50}")
    print(f"  LogAnalyzer PRO — {params:,} params")
    print(f"  Data: {n:,} | Device: {DEVICE} | AMP: {USE_AMP}")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"{'=' * 50}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        print(f"\n>>> Epoch {epoch + 1}/{args.epochs} — training...")
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        n_batches = len(train_loader)

        for batch_idx, (tokens, labels) in enumerate(train_loader):
            tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            if USE_AMP:
                with torch.amp.autocast(DEVICE.type):
                    outputs = model(tokens)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(tokens)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()
            _, preds = outputs.max(1)
            train_total += labels.size(0)
            train_correct += preds.eq(labels).sum().item()

            # Log progress
            if batch_idx == 0:
                print(f"  Batch 1 done — loss={loss.item():.4f}")
            elif (
                batch_idx < 5
                or (batch_idx + 1) % max(1, n_batches // 100) == 0
                or batch_idx == n_batches - 1
            ):
                pct = (batch_idx + 1) / n_batches * 100
                avg_loss = train_loss / (batch_idx + 1)
                avg_acc = 100 * train_correct / max(train_total, 1)
                elapsed = time.time() - t0
                remain = (elapsed / (batch_idx + 1)) * (n_batches - batch_idx - 1)
                print(
                    f"  [{pct:4.1f}%] loss={avg_loss:.4f} acc={avg_acc:.2f}% | {elapsed:.0f}s elapsed / {remain:.0f}s remain"
                )

        print()
        train_loss /= n_batches
        train_acc = 100.0 * train_correct / train_total

        print(f"  >> Validating...")
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for v_idx, (tokens, labels) in enumerate(val_loader):
                if v_idx % max(1, len(val_loader) // 5) == 0:
                    print(f"    validation {v_idx + 1}/{len(val_loader)}", end="\r")
                tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
                if USE_AMP:
                    with torch.amp.autocast(DEVICE.type):
                        outputs = model(tokens)
                        loss = criterion(outputs, labels)
                else:
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
        elap = time.time() - t0
        log.info(
            f"E{epoch + 1}/{args.epochs} | Train: {train_loss:.4f} {train_acc:.2f}% | Val: {val_loss:.4f} {val_acc:.2f}% | {elap:.0f}s"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "best_val_loss": best_val_loss,
            },
            ckpt_path,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_DIR / "best_model.pt")
            log.info(f"  > Best model saved ({val_loss:.4f})")

    # ── Export inference ──
    print(f"\n{'=' * 50}")
    print("  TRAINING COMPLETE — Exporting inference artifacts")
    print(f"{'=' * 50}")

    torch.save(model.state_dict(), SAVE_DIR / "inference_model.pt")
    cfg = {
        "max_seq": args.max_seq,
        "num_classes": 2,
        "params": params,
        "vocab_size": VOCAB_SIZE,
        "d_model": 256,
        "n_layers": 6,
        "n_heads": 8,
        "n_kv_heads": 4,
        "d_ff": 1024,
    }
    with open(SAVE_DIR / "model_config.json", "w") as f:
        json.dump(cfg, f)

    try:
        model.eval()
        dummy = torch.randint(0, VOCAB_SIZE, (1, args.max_seq)).to(DEVICE)
        torch.onnx.export(
            model,
            dummy,
            SAVE_DIR / "model.onnx",
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )
        log.info("ONNX exported")
    except Exception as e:
        log.warning(f"ONNX skipped: {e}")

    # Log model size
    pt_size = os.path.getsize(SAVE_DIR / "inference_model.pt") / 1024 / 1024
    onnx_size = (
        os.path.getsize(SAVE_DIR / "model.onnx") / 1024 / 1024
        if (SAVE_DIR / "model.onnx").exists()
        else 0
    )
    log.info(f"Models saved to {SAVE_DIR}")
    log.info(f"  inference_model.pt  = {pt_size:.1f} MB")
    log.info(f"  model.onnx          = {onnx_size:.1f} MB")
    log.info(f"  model_config.json   = config")

    best_idx = int(np.argmin(history["val_loss"]))
    print(f"\n  Final Results:")
    print(
        f"  Best Val:  loss={history['val_loss'][best_idx]:.4f} acc={history['val_acc'][best_idx]:.2f}% (epoch {best_idx + 1})"
    )
    print(
        f"  Final Val: loss={history['val_loss'][-1]:.4f} acc={history['val_acc'][-1]:.2f}%"
    )
    print(f"  Params: {params:,}")
    print(
        f"  Inference: inference_model.pt (~{pt_size:.0f}MB) or model.onnx (~{onnx_size:.0f}MB)"
    )
    print(f"{'=' * 50}")


if __name__ == "__main__":
    train()
