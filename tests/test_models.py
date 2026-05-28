"""Tests cho models module."""

import torch
import pytest

from src.models.base import MLP


class TestMLP:
    def test_forward_shape(self) -> None:
        model = MLP(input_dim=10, hidden_dims=(20, 10), output_dim=5)
        x = torch.randn(4, 10)
        output = model(x)
        assert output.shape == (4, 5)

    def test_count_parameters(self) -> None:
        model = MLP(input_dim=784, hidden_dims=(256, 128), output_dim=10)
        n = model.count_parameters()
        assert n > 0

    def test_save_load_checkpoint(self, tmp_path: pytest.TempPathFactory) -> None:
        model = MLP(input_dim=10, hidden_dims=(8,), output_dim=3)
        path = tmp_path / "test_model.pt"
        model.save_checkpoint(path, epoch=5)
        assert path.exists()

        model2 = MLP(input_dim=10, hidden_dims=(8,), output_dim=3)
        checkpoint = model2.load_checkpoint(path)
        assert checkpoint["epoch"] == 5
