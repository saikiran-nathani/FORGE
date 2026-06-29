"""Shared test fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def sample_classification_df():
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 28, 33, 38, 42],
        "income": [50000.0, 60000.0, 75000.0, 80000.0, 90000.0, 55000.0, 62000.0, 71000.0, 85000.0, 48000.0],
        "city": ["NYC", "LA", "NYC", "Chicago", "LA", "NYC", "Chicago", "LA", "NYC", "Chicago"],
        "churn": [0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
    })


@pytest.fixture
def sample_regression_df():
    return pd.DataFrame({
        "sqft": [1000, 1500, 2000, 2500, 3000, 1200, 1800, 2200, 2800, 1600],
        "bedrooms": [2, 3, 3, 4, 4, 2, 3, 3, 4, 3],
        "age": [10, 5, 20, 15, 2, 8, 12, 18, 3, 7],
        "price": [200000, 300000, 350000, 450000, 550000, 250000, 320000, 380000, 520000, 290000],
    })
