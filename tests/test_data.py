"""Tests cho data module."""

import pandas as pd
import pytest

from src.data.loader import DataLoader
from src.data.preprocessor import Preprocessor


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [5, 6, 7, 8, 9],
            "label": [0, 1, 0, 1, 0],
        }
    )


class TestPreprocessor:
    def test_remove_duplicates(self, sample_df: pd.DataFrame) -> None:
        preprocessor = Preprocessor()
        df_with_dup = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
        result = preprocessor.remove_duplicates(df_with_dup)
        assert len(result) == len(sample_df)

    def test_handle_missing_values_mean(self) -> None:
        preprocessor = Preprocessor()
        df = pd.DataFrame({"a": [1, 2, None, 4]})
        result = preprocessor.handle_missing_values(df, strategy="mean")
        assert result.isnull().sum().sum() == 0
        assert result.loc[2, "a"] == pytest.approx(7 / 3)

    def test_train_val_test_split(self) -> None:
        preprocessor = Preprocessor()
        import numpy as np

        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        X_train, X_val, X_test, y_train, y_val, y_test = (
            preprocessor.train_val_test_split(X, y, test_size=0.2, val_size=0.1)
        )
        # 100 * 0.2 = 20 test, còn 80. val_ratio = 0.1 / 0.8 = 0.125
        # 80 * 0.125 = 10 val, 80 - 10 = 70 train
        assert len(X_train) == 70
        assert len(X_val) == 10
        assert len(X_test) == 20
