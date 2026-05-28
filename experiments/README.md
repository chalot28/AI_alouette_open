# 🧪 Experiments

Thư mục này dùng để theo dõi các thí nghiệm huấn luyện.

## Cấu trúc gợi ý

```
experiments/
├── experiment_001/
│   ├── config.yaml
│   ├── metrics.json
│   ├── model.pt
│   └── notes.md
├── experiment_002/
│   └── ...
└── README.md
```

## Công cụ tracking (gợi ý)

- **MLflow**: `mlflow run .`
- **TensorBoard**: `tensorboard --logdir logs/tensorboard`
- **Weights & Biases**: `wandb init`

Ghi chép mỗi experiment gồm:
- Mục tiêu
- Hyperparameters
- Kết quả metrics
- Nhận xét / bài học
