"""
Script chạy inference với model đã trained.

Usage:
    python scripts/run_inference.py --model models/best_model.pt --input data/raw/test.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config.config import config
from src.data.preprocessor import Preprocessor
from src.models.base import MLP
from src.models.predict import Predictor
from src.utils.helpers import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alouette AI — Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to input data (npy/csv)")
    parser.add_argument("--output", type=str, default="predictions.json", help="Output path")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load model ──
    logger.info(f"Loading model từ: {args.model}")
    model = MLP(
        input_dim=20,  # Sẽ khớp với dim của data
        hidden_dims=(128, 64),
        output_dim=2,
    )
    model.load_checkpoint(args.model, map_location="cpu")
    logger.info(f"\n{model.summary()}")

    # ── Load data ──
    input_path = Path(args.input)
    if input_path.suffix == ".npy":
        X = np.load(input_path)
    elif input_path.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(input_path)
        # Giả sử cột cuối là label (optional)
        if "label" in df.columns:
            df = df.drop(columns=["label"])
        X = df.values
    else:
        raise ValueError(f"Định dạng '{input_path.suffix}' không hỗ trợ.")

    # ── Scale ──
    preprocessor = Preprocessor()
    X = preprocessor.scale_features(X, method="standard", fit=True)

    # ── Predict ──
    predictor = Predictor(model, device="cpu")
    preds = predictor.predict(X, batch_size=args.batch_size, return_proba=False)
    probs = predictor.predict(X, batch_size=args.batch_size, return_proba=True)

    # ── Save ──
    results = {
        "predictions": preds.tolist(),
        "probabilities": probs.tolist(),
    }
    save_json(results, args.output)
    logger.info(f"Đã lưu kết quả tại: {args.output}")
    logger.info(f"Dự đoán {len(preds)} samples. Phân bố: {np.bincount(preds).tolist()}")


if __name__ == "__main__":
    main()
