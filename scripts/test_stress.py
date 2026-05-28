"""
Kiểm tra model: tài nguyên + test lỗi cực khó.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from scripts.train_pro import CHAR2IDX, LogAnalyzerPro

model = LogAnalyzerPro(max_seq=1024)
model.load_state_dict(torch.load("models/best_model.pt", map_location="cpu"))
model.eval()
print(f"Model: {model.count_params():,} params\n")


def tokenize(text):
    arr = np.zeros(1024, dtype=np.int64)
    for j, ch in enumerate(text[:1024]):
        arr[j] = CHAR2IDX.get(ch, 0)
    return torch.tensor(arr).unsqueeze(0)


def predict(text):
    with torch.no_grad():
        logits = model(tokenize(text))
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax().item()
        return ("ERROR" if pred else "NORMAL", probs[0][pred].item())


# ═══════════════════════════════════════
# TEST RESOURCE
# ═══════════════════════════════════════
print("═" * 60)
print("📊 TEST TÀI NGUYÊN")
print("═" * 60)

import psutil

proc = psutil.Process()

# RAM trước
ram_before = proc.memory_info().rss / 1024 / 1024
print(f"RAM trước khi load model: {ram_before:.0f} MB")

# RAM sau khi load model (đã load ở trên)
ram_after = proc.memory_info().rss / 1024 / 1024
print(f"RAM sau khi load model:  {ram_after:.0f} MB")
print(f"Model chiếm:             {ram_after - ram_before:.0f} MB")

# CPU + throughput
sample = "ERROR,2026-05-28,UserService,NullPointerException,user is null userId=12345"
tokens = tokenize(sample)

# Warm-up
for _ in range(10):
    model(tokens)

# Benchmark throughput
N = 100
start = time.time()
for _ in range(N):
    model(tokens)
elapsed = time.time() - start
throughput = N / elapsed
print(f"\nThroughput: {throughput:.0f} inferences/s")
print(f"Thời gian 1 lần: {elapsed / N * 1000:.1f} ms")
print(f"1000 logs xử lý trong: {1000 / throughput:.1f}s")

# CPU usage
cpu_pct = proc.cpu_percent(interval=1)
print(f"CPU usage: {cpu_pct:.0f}% (1 core)")

# ═══════════════════════════════════════
# TEST EDGE CASES CỰC KHÓ
# ═══════════════════════════════════════
print("\n" + "═" * 60)
print("🧠 TEST EDGE CASES — LỖI CỰC KHÓ")
print("═" * 60)

tests = [
    # (Tên, text, expected)
    # ── LỖI ĐÁNH LỪA (adversarial) ──
    (
        "LỪA: error trong normal",
        "INFO,2026-05-28,TestService,This is a test with error handling that works fine code=200",
        "NORMAL",
    ),
    (
        "LỪA: exception trong normal",
        "INFO,2026-05-28,TestService,Exception handling test passed successfully no errors found",
        "NORMAL",
    ),
    (
        "LỪA: null trong normal",
        "DEBUG,2026-05-28,TestService,Setting null value for optional field defaultValue=0",
        "NORMAL",
    ),
    (
        "LỪA: failed thành công",
        "INFO,2026-05-28,TestService,Previous failed attempt recovered successfully retryCount=3",
        "NORMAL",
    ),
    (
        "LỪA: timeout bình thường",
        "INFO,2026-05-28,TestService,Connection timeout configured to 30s for external service",
        "NORMAL",
    ),
    (
        "LỪA: denied nhưng ok",
        "INFO,2026-05-28,AuthService,Access denied handler registered successfully for role=admin",
        "NORMAL",
    ),
    # ── NORMAL ĐÓNG GIẢ LỖI ──
    (
        "NGỤY TRANG: lỗi nhẹ",
        "ERROR,2026-05-28,TestService,A minor warning was raised but system is healthy uptime=99.9%",
        "ERROR",
    ),
    (
        "NGỤY TRANG: exception test",
        "ERROR,2026-05-28,TestService,Unit test for exception handling passed all assertions",
        "ERROR",
    ),
    (
        "NGỤY TRANG: fake error log",
        "ERROR,2026-05-28,TestService,This is a test log with no actual problem everything is fine",
        "ERROR",
    ),
    (
        "NGỤY TRANG: null test",
        "ERROR,2026-05-28,TestService,Testing null pointer scenario for code coverage report",
        "ERROR",
    ),
    # ── TIN NHẮN NGẮN (khó vì ít context) ──
    ("SIÊU NGẮN: error", "ERROR,2026-05-28,TEST,err", "ERROR"),
    ("SIÊU NGẮN: ok", "INFO,2026-05-28,TEST,ok", "NORMAL"),
    ("SIÊU NGẮN: x", "ERROR,2026-05-28,X,X", "ERROR"),
    # ── TIN NHẮN DÀI ──
    (
        "RẤT DÀI: normal",
        "DEBUG,2026-05-28,UserService," + "a" * 800 + "successful" + "b" * 200,
        "NORMAL",
    ),
    (
        "RẤT DÀI: error",
        "ERROR,2026-05-28,TestService," + "x" * 800 + "timeout" + "y" * 200,
        "ERROR",
    ),
    # ── LEVEL SAI ──
    (
        "LEVEL SAI: CRITICAL",
        "CRITICAL,2026-05-28,TestService,Database connection lost critical error",
        "ERROR",
    ),
    (
        "LEVEL LẠ: TRACE",
        "TRACE,2026-05-28,TestService,Verbose debug output for method entry",
        "NORMAL",
    ),
    (
        "LEVEL LẠ: NOTICE",
        "NOTICE,2026-05-28,TestService,System will restart for maintenance at 02:00",
        "NORMAL",
    ),
    # ── KÝ TỰ ĐẶC BIỆT ──
    (
        "KÝ TỰ LẠ: unicode",
        "ERROR,2026-05-28,支付服务,NullPointerException 用户登录失败 userId=你好",
        "ERROR",
    ),
    (
        "KÝ TỰ LẠ: emoji",
        "INFO,2026-05-28,ChatService,User sent a message 😊✅🔥 messageId=123",
        "NORMAL",
    ),
    (
        "KÝ TỰ LẠ: html",
        'ERROR,2026-05-28,WebhookHandler,XSS payload <script>alert("xss")</script> detected',
        "ERROR",
    ),
    # ── NHIỄU (random string) ──
    (
        "NHIỄU: error level random",
        "ERROR,2026-05-28,TestService,aksjdhasjkdhaksjhd",
        "ERROR",
    ),
    (
        "NHIỄU: info level random",
        "INFO,2026-05-28,TestService,xcnvbmxncvbmxcvbm",
        "NORMAL",
    ),
    # ── FORMAT KHÁC ──
    (
        "FORMAT LẠ: no service",
        "ERROR,2026-05-28,,NullPointerException,user is null",
        "ERROR",
    ),
    (
        "FORMAT LẠ: no timestamp",
        "ERROR,2026-05-28,TestService,NullPointerException,user is null",
        "ERROR",
    ),
    (
        "FORMAT LẠ: JSON",
        'ERROR,2026-05-28,API,HttpClientErrorException,{"code":500,"msg":"internal error"}',
        "ERROR",
    ),
    (
        "FORMAT LẠ: JSON normal",
        'INFO,2026-05-28,API,{"status":"ok","code":200}',
        "NORMAL",
    ),
]

correct = 0
print(f"\n{'#':<3} {'Test':<35} {'Expected':<10} {'Got':<10} {'Conf':<8} Result")
print("-" * 80)

for i, (name, text, expected) in enumerate(tests, 1):
    pred, conf = predict(text)
    ok = "✅" if pred == expected else "❌"
    if ok == "✅":
        correct += 1
    print(f"{i:<3} {name:<35} {expected:<10} {pred:<10} {conf:.2%}    {ok}")

print(f"\nKết quả: {correct}/{len(tests)} = {correct / len(tests) * 100:.0f}%")
print(f"RAM: {ram_after:.0f} MB | Throughput: {throughput:.0f} inferences/s")
