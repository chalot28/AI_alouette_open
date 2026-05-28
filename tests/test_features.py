"""Tests cho features module."""

import pandas as pd
import pytest

from src.features.builder import FeatureBuilder


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "x1": [1, 2, 3],
        "x2": [4, 5, 6],
        "date": ["2024-01-01", "2024-06-15", "2024-12-31"],
    })


class TestFeatureBuilder:
    def test_add_interaction(self, sample_df: pd.DataFrame) -> None:
        builder = FeatureBuilder(sample_df)
        result = builder.add_interaction("x1", "x2", "multiply").build()
        assert "x1_multiply_x2" in result.columns
        assert result["x1_multiply_x2"].tolist() == [4, 10, 18]

    def test_add_datetime_features(self, sample_df: pd.DataFrame) -> None:
        builder = FeatureBuilder(sample_df)
        result = builder.add_datetime_features("date", features=["year", "month"]).build()
        assert "date_year" in result.columns
        assert result["date_year"].tolist() == [2024, 2024, 2024]
        assert result["date_month"].tolist() == [1, 6, 12]
