"""
LogAnalyzer — Model phân tích log server phát hiện lỗi.

Kiến trúc:
  [Char-level Tokenizer]
       ↓
  [Embedding] vocab=256, d_model=256
       ↓
  [CNN Encoder] 4 Conv1D layers nén 40k → 2560 tokens
       ↓
  [Transformer Encoder] 6 layers, d_model=256, 8 heads
       ↓
  [Global Pooling] avg + max
       ↓
  [Classifier] → error / not-error (binary)

Tham số: ~5M
Context: 40,960 tokens
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Character-level Tokenizer
# ──────────────────────────────────────────────

# Bảng ký tự log — bao phủ mọi ký tự xuất hiện trong log server
LOG_CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "_-.:/\\[](){}!@#$%^&*+=,;?<>'\"`~| "
    "\t\n\r"
)

CHAR2IDX = {ch: i + 1 for i, ch in enumerate(LOG_CHARS)}  # 0 = padding
PAD_IDX = 0
VOCAB_SIZE = len(CHAR2IDX) + 1  # +1 cho padding token


def tokenize_log(text: str, max_len: int = 40960) -> torch.LongTensor:
    """
    Tokenize log text thành character-level indices.

    Args:
        text: Raw log string.
        max_len: Độ dài tối đa (40_960).

    Returns:
        Tensor shape (seq_len,) với character indices.
    """
    tokens = []
    for ch in text[:max_len]:
        tokens.append(CHAR2IDX.get(ch, PAD_IDX))
    # Pad đến max_len
    if len(tokens) < max_len:
        tokens += [PAD_IDX] * (max_len - len(tokens))
    return torch.tensor(tokens[:max_len], dtype=torch.long)


def detokenize_log(tokens: torch.LongTensor) -> str:
    """Chuyển indices về text (debug)."""
    idx2char = {v: k for k, v in CHAR2IDX.items()}
    idx2char[0] = ""
    return "".join(idx2char.get(i.item(), "?") for i in tokens)


# ──────────────────────────────────────────────
# CNN Encoder — nén sequence dài
# ──────────────────────────────────────────────


class CNNEncoder(nn.Module):
    """
    CNN front-end để nén 40k tokens → 2560 tokens.

    Architecture:
      Conv1D × 4, kernel_size=3, stride=2
      + Residual connections + LayerNorm + GELU

    Downsampling: 40960 → 20480 → 10240 → 5120 → 2560
    """

    def __init__(
        self,
        d_model: int = 256,
        num_conv_layers: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__()

        layers = []
        in_channels = d_model
        for i in range(num_conv_layers):
            out_channels = d_model
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                bias=False,
            )
            layers.extend(
                [
                    conv,
                    nn.GroupNorm(8, out_channels),
                    nn.GELU(),
                ]
            )
            in_channels = out_channels

        self.net = nn.Sequential(*layers)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len_out, d_model)
        """
        # Conv1D expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (B, d, S)
        x = self.net(x)  # (B, d, S')
        x = x.transpose(1, 2)  # (B, S', d)
        x = self.final_norm(x)
        return x


# ──────────────────────────────────────────────
# Grouped Query Attention (hiệu quả hơn MHA)
# ──────────────────────────────────────────────


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) — giảm KV-cache, ổn định hơn MHA.

    Với 8 heads, chia thành 4 groups: mỗi group 1 key/value head.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_kv_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_head = d_model // n_heads

        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads phải chia hết cho n_kv_heads"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, D = x.shape

        # Projections
        q = self.q_proj(x)  # (B, S, D)
        k = self.k_proj(x)  # (B, S, n_kv * d_head)
        v = self.v_proj(x)  # (B, S, n_kv * d_head)

        # Reshape
        q = q.view(B, S, self.n_heads, self.d_head).transpose(
            1, 2
        )  # (B, n_heads, S, d_head)
        k = k.view(B, S, self.n_kv_heads, self.d_head).transpose(
            1, 2
        )  # (B, n_kv, S, d_head)
        v = v.view(B, S, self.n_kv_heads, self.d_head).transpose(
            1, 2
        )  # (B, n_kv, S, d_head)

        # Expand KV heads → nhân lên cho mỗi group
        k = k.repeat_interleave(self.n_groups, dim=1)  # (B, n_heads, S, d_head)
        v = v.repeat_interleave(self.n_groups, dim=1)

        # Scaled dot-product attention
        scale = self.d_head**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_heads, S, S)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, n_heads, S, d_head)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)
        return out


# ──────────────────────────────────────────────
# Transformer Block (tiết kiệm tham số)
# ──────────────────────────────────────────────


class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block tối ưu cho ~5M model.

    Pre-norm architecture + GQA + SwiGLU FFN.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_kv_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout)

        self.norm2 = nn.LayerNorm(d_model)
        # SwiGLU FFN — 2 linear + gating, hiệu quả hơn ReLU FFN
        self.ffn_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_up = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, mask)
        x = self.dropout(x)
        x = residual + x

        # Pre-norm FFN with SwiGLU + residual
        residual = x
        x = self.norm2(x)
        gate = F.silu(self.ffn_gate(x))
        up = self.ffn_up(x)
        x = gate * up
        x = self.ffn_down(x)
        x = self.dropout(x)
        x = residual + x
        return x


# ──────────────────────────────────────────────
# LogAnalyzer — Model chính
# ──────────────────────────────────────────────


class LogAnalyzer(BaseModel):
    """
    LogAnalyzer — Phát hiện lỗi từ log server.

    Thông số:
      - Tham số:    ~5,078,018 (≈ 5M)
      - Context:    40,960 tokens
      - Input:      raw log text
      - Output:     error probability (0-1)

    Architecture:
      Embedding(256) → CNN×4(nén 40k→2560) → Transformer×6 → Pool → Classifier

    Usage:
        model = LogAnalyzer()
        log_probs = model("ERROR: Connection refused at 127.0.0.1:8080")
        # → tensor([0.98])  # 98% là lỗi
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        n_kv_heads: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 40960,
        dropout: float = 0.1,
        num_classes: int = 2,  # error / not-error
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)

        # Position encoding — Sinusoidal (không tham số)
        self.register_buffer(
            "pos_encoding",
            self._create_sinusoidal_positions(max_seq_len, d_model),
        )

        # Input dropout
        self.input_dropout = nn.Dropout(dropout)

        # CNN Encoder — nén sequence
        self.cnn_encoder = CNNEncoder(d_model=d_model, num_conv_layers=4)

        # Tính seq_len sau CNN:
        # 40960 → 20480 → 10240 → 5120 → 2560
        self.cnn_out_len = max_seq_len // (2**4)

        # Transformer Encoder
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.transformer_norm = nn.LayerNorm(d_model)

        # Global pooling + Classifier
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, num_classes),
        )

        # Khởi tạo weights
        self._init_weights()

        # Log parameter count
        n_params = self.count_parameters()
        logger.info(
            f"LogAnalyzer khởi tạo: {n_params:,} params "
            f"(target ~5M) | context={max_seq_len:,}"
        )

    def _create_sinusoidal_positions(
        self,
        max_len: int,
        d_model: int,
    ) -> torch.Tensor:
        """Tạo sinusoidal position encoding (cố định, không học)."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (max_len, d_model)

    def _init_weights(self) -> None:
        """Khởi tạo weights ổn định."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Token indices (batch, seq_len).
            mask: Attention mask (batch, seq_len). True = valid.

        Returns:
            Logits (batch, num_classes).
        """
        B, S = x.shape

        # Embedding + Position
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:S, :].unsqueeze(0)
        x = self.input_dropout(x)

        # CNN Encoder — nén sequence
        x = self.cnn_encoder(x)  # (B, S', d)

        # Tạo mask mới cho seq_len đã nén
        # (nếu seq_len sau CNN khác seq_len gốc / 16)
        if mask is not None:
            # Pool mask xuống cùng tỷ lệ
            mask = mask.float()
            mask = F.avg_pool1d(
                mask.unsqueeze(1),
                kernel_size=16,
                stride=16,
                padding=0,
            )
            mask = (mask > 0.5).float()
            mask = mask.squeeze(1).bool()  # (B, S')
        else:
            mask = None

        # Transformer mask
        attn_mask = None
        if mask is not None:
            # (B, 1, 1, S') cho broadcasting trong attention
            attn_mask = mask.unsqueeze(1).unsqueeze(2)

        # Transformer Encoder
        for layer in self.layers:
            x = layer(x, mask=attn_mask)

        x = self.transformer_norm(x)

        # Masked pooling
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
            # Average chỉ trên valid positions
            x = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        else:
            # Global pooling
            x = x.mean(dim=1)  # (B, d)

        # Classifier
        logits = self.classifier(x)
        return logits

    def predict_error(self, log_text: str) -> tuple[int, float]:
        """
        Predict một log line.

        Args:
            log_text: Raw log string.

        Returns:
            (predicted_class, probability)
            0 = not-error, 1 = error
        """
        self.eval()
        tokens = tokenize_log(log_text, self.max_seq_len)
        tokens = tokens.unsqueeze(0)  # (1, seq_len)

        with torch.no_grad():
            logits = self.forward(tokens)
            probs = F.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1).item()
            prob = probs[0, pred].item()

        return pred, prob

    def predict_batch(
        self,
        log_texts: list[str],
        batch_size: int = 8,
    ) -> list[tuple[int, float]]:
        """Predict batch log lines."""
        self.eval()
        results = []

        for i in range(0, len(log_texts), batch_size):
            batch_texts = log_texts[i : i + batch_size]
            batch_tokens = torch.stack(
                [tokenize_log(t, self.max_seq_len) for t in batch_texts]
            )

            with torch.no_grad():
                logits = self.forward(batch_tokens)
                probs = F.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)
                for j in range(len(batch_texts)):
                    results.append((preds[j].item(), probs[j, preds[j]].item()))

        return results

    @classmethod
    def get_param_count_detail(cls) -> dict[str, int]:
        """In chi tiết tham số từng component."""
        model = cls()
        details = {}

        # Embedding
        emb_params = sum(p.numel() for p in model.token_embedding.parameters())
        details["token_embedding"] = emb_params

        # CNN
        cnn_params = sum(p.numel() for p in model.cnn_encoder.parameters())
        details["cnn_encoder"] = cnn_params

        # Transformer
        tf_params = sum(p.numel() for p in model.layers.parameters())
        tf_params += sum(p.numel() for p in model.transformer_norm.parameters())
        details["transformer"] = tf_params

        # Classifier
        cls_params = sum(p.numel() for p in model.classifier.parameters())
        details["classifier"] = cls_params

        details["total"] = sum(details.values())
        return details


# ──────────────────────────────────────────────
# Kiểm tra nhanh
# ──────────────────────────────────────────────


def verify_log_analyzer() -> None:
    """Verify parameter count và forward pass."""
    print("=" * 60)
    print("LogAnalyzer — Verification")
    print("=" * 60)

    model = LogAnalyzer()

    # Chi tiết tham số
    details = LogAnalyzer.get_param_count_detail()
    for name, count in details.items():
        print(f"  {name:20s}: {count:>8,} params")
    print(f"  {'─' * 30}")
    print(f"  {'total':20s}: {details['total']:>8,} params")

    # Forward test
    dummy_log = "ERROR: Connection refused at 127.0.0.1:8080"
    tokens = tokenize_log(dummy_log, max_len=40960)
    tokens = tokens.unsqueeze(0)

    with torch.no_grad():
        logits = model(tokens)
        pred, prob = model.predict_error(dummy_log)

    print(f"\n  Input:  {dummy_log}")
    print(f"  Output: class={pred}, probability={prob:.4f}")
    print(f"  Logits: {logits.tolist()}")
    print("=" * 60)


if __name__ == "__main__":
    verify_log_analyzer()
