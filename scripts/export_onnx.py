"""
Export LogAnalyzerPro → ONNX
Chạy không cần PyTorch, siêu nhẹ (~15MB).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from scripts.train_pro import VOCAB_SIZE, LogAnalyzerPro

MODEL_PATH = "models/best_model.pt"
ONNX_PATH = "models/log_analyzer.onnx"
CONFIG_PATH = "models/model_config.json"
MAX_SEQ = 1024

# Load model
print("Loading model...")
model = LogAnalyzerPro(max_seq=MAX_SEQ)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print(f"Params: {model.count_params():,}")

# Save config
config = {
    "max_seq": MAX_SEQ,
    "num_classes": 2,
    "params": model.count_params(),
    "vocab_size": VOCAB_SIZE,
    "d_model": 256,
    "n_layers": 6,
    "n_heads": 8,
    "n_kv_heads": 4,
    "d_ff": 1024,
}
with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)
print(f"Config saved: {CONFIG_PATH}")

# Export ONNX bằng chế độ cũ (ổn định, giữ nguyên weights)
print("Exporting to ONNX...")
model.eval()

# Forward một lần để warm-up
with torch.no_grad():
    dummy = torch.randint(0, VOCAB_SIZE, (1, MAX_SEQ))
    _ = model(dummy)

# Export với dynamo=False (chế độ cũ)
torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=18,
    do_constant_folding=True,
    export_params=True,
)
print(f"ONNX saved: {ONNX_PATH}")

# Verify
import onnx

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
print(
    f"✅ Verified! Input: {onnx_model.graph.input[0].name}, Output: {[o.name for o in onnx_model.graph.output]}"
)

size_mb = Path(ONNX_PATH).stat().st_size / 1024 / 1024
print(f"Size: {size_mb:.1f} MB (should be ~15-24MB)")
