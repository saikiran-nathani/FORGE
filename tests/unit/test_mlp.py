"""Basic MLP model smoke test."""

import numpy as np
import pandas as pd

from forge.training.deep_learning.nn_wrappers import TabularMLPClassifier
from forge.training.task_router import TaskType
from forge.training.deep_learning.models import MLPModel


def test_mlp_classifier_trains():
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(100, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.randint(0, 2, 100))

    model = TabularMLPClassifier(epochs=5, patience=2, batch_size=32)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 100


def test_mlp_model_search_space():
    import optuna
    m = MLPModel(TaskType.BINARY_CLASSIFICATION)
    trial = optuna.trial.FixedTrial({"hidden_dim": 64, "n_layers": 2, "dropout": 0.2,
                                      "lr": 0.001, "weight_decay": 1e-4, "batch_size": 64,
                                      "use_batch_norm": True})
    params = m.get_search_space(trial)
    assert "hidden_dim" in params
