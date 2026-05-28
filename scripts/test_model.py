"""
Test LogAnalyzerPro — thử độ thông minh của model.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from scripts.train_pro import CHAR2IDX, VOCAB_SIZE, LogAnalyzerPro

MODEL_PATH = "models/best_model.pt"
MAX_SEQ = 1024

# Load model
model = LogAnalyzerPro(max_seq=MAX_SEQ)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print(f"Model: {model.count_params():,} params\n")


# Tokenize
def tokenize(text):
    arr = np.zeros(MAX_SEQ, dtype=np.int64)
    for j, ch in enumerate(text[:MAX_SEQ]):
        arr[j] = CHAR2IDX.get(ch, 0)
    return torch.tensor(arr).unsqueeze(0)


# Test samples
tests = [
    # ── ERROR ──
    (
        "ERROR",
        "NullPointerException",
        "Cannot invoke getId() because user is null userId=12345",
    ),
    (
        "FATAL",
        "OutOfMemoryError",
        "Java heap space exhausted during export requestedRows=5000000",
    ),
    (
        "ERROR",
        "SQLException",
        "Deadlock detected when updating order orderId=ord_99999",
    ),
    (
        "ERROR",
        "TimeoutException",
        "Payment gateway did not respond within 30s orderId=ord_88888",
    ),
    (
        "ERROR",
        "ConnectionRefusedException",
        "Cannot connect to MySQL at host=db-primary-1.internal port=3306",
    ),
    (
        "ERROR",
        "HttpClientErrorException",
        "502 Bad Gateway from payment-service/authorize upstreamTimeout=45s",
    ),
    (
        "ERROR",
        "SecurityException",
        "SQL injection attempt detected input=1; DROP TABLE users userId=77777",
    ),
    (
        "ERROR",
        "IllegalArgumentException",
        "Negative amount not allowed amount=-5000 userId=11111",
    ),
    # ── NORMAL ──
    ("INFO", None, "User login successful userId=54321 ip=192.168.1.1"),
    (
        "INFO",
        None,
        "Order created successfully orderId=ord_12345 userId=54321 total=299",
    ),
    (
        "INFO",
        None,
        "Payment processed transactionId=tx_98765 amount=50000 status=success",
    ),
    ("DEBUG", None, "Processing GET /api/v2/users/12345"),
    ("INFO", None, "Cache hit for key=user:54321 latency=3ms"),
    ("WARN", None, "Memory usage high heap=85% used=2048MB max=4096MB"),
    ("INFO", None, "Search results query=laptop hits=42 time=150ms"),
    ("INFO", None, "Backup completed type=full size=2048MB duration=1200s"),
]

print(f"{'Level':<8} {'Expected':<8} {'Predicted':<8} {'Conf':<8}  Message")
print("-" * 80)

correct = 0
for level, exc, msg in tests:
    is_error = 1 if level in ("ERROR", "FATAL") else 0
    text = (
        f"{level},2026-05-28 12:00:00,TestService,{exc},{msg}"
        if exc
        else f"{level},2026-05-28 12:00:00,TestService,{msg}"
    )

    with torch.no_grad():
        logits = model(tokenize(text))
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax().item()
        conf = probs[0, pred].item()

    expected = "ERROR" if is_error == 1 else "NORMAL"
    predicted = "ERROR" if pred == 1 else "NORMAL"
    is_correct = "✅" if pred == is_error else "❌"

    if is_correct == "✅":
        correct += 1

    print(
        f"{level:<8} {expected:<8} {predicted:<8} {conf:.2%}    {is_correct} {msg[:50]}"
    )

print(f"\nAccuracy: {correct}/{len(tests)} = {correct / len(tests) * 100:.0f}%")
