"""Sklearn-compatible PyTorch tabular models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from forge.training.deep_learning.trainer import DLConfig, DLTrainer


class MLPNetwork(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class TabularMLP(BaseEstimator):
    """Sklearn wrapper for tabular MLP."""

    def __init__(
        self,
        task_type: str = "classification",
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        batch_size: int = 128,
        patience: int = 8,
        random_state: int = 42,
        use_batch_norm: bool = True,
    ):
        self.task_type = task_type
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.use_batch_norm = use_batch_norm
        self.model_: nn.Module | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int = 0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]
        is_regression = self.task_type == "regression"

        if is_regression:
            n_outputs = 1
            y_train = y.astype(np.float32)
        else:
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)
            n_outputs = 1 if n_classes == 2 else n_classes
            y_train = y.astype(np.int64)

        network = MLPNetwork(
            self.n_features_in_, n_outputs,
            self.hidden_dim, self.n_layers, self.dropout, self.use_batch_norm,
        )
        config = DLConfig(
            lr=self.lr, weight_decay=self.weight_decay, epochs=self.epochs,
            batch_size=self.batch_size, patience=self.patience, random_state=self.random_state,
        )
        self.model_ = DLTrainer(config, self.task_type).train(network, X, y_train)
        return self

    def predict(self, X):
        proba = self._forward(X)
        if self.task_type == "regression":
            return proba.squeeze()
        if proba.shape[-1] == 1:
            return (proba.squeeze() >= 0.5).astype(int)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        if self.task_type == "regression":
            raise AttributeError("predict_proba not available for regression")
        proba = self._forward(X)
        if proba.shape[-1] == 1:
            p = proba.squeeze()
            return np.column_stack([1 - p, p])
        return proba

    def _forward(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        device = next(self.model_.parameters()).device
        self.model_.eval()
        with torch.no_grad():
            out = self.model_(torch.tensor(X, device=device))
            if self.task_type == "regression":
                return out.cpu().numpy()
            if out.shape[-1] == 1:
                return torch.sigmoid(out).cpu().numpy()
            return torch.softmax(out, dim=-1).cpu().numpy()


class TabularMLPClassifier(TabularMLP, ClassifierMixin):
    def __init__(self, **kwargs):
        super().__init__(task_type="classification", **kwargs)


class TabularMLPRegressor(TabularMLP, RegressorMixin):
    def __init__(self, **kwargs):
        super().__init__(task_type="regression", **kwargs)


class TabTransformerNetwork(nn.Module):
    """Simplified TabTransformer for all-numerical tabular data."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_outputs),
        )
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x.unsqueeze(-1))
        cls = self.cls.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0])


class TabularTransformer(TabularMLP):
    """Sklearn wrapper using TabTransformerNetwork."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]
        is_regression = self.task_type == "regression"

        if is_regression:
            n_outputs = 1
            y_train = y.astype(np.float32)
        else:
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)
            n_outputs = 1 if n_classes == 2 else n_classes
            y_train = y.astype(np.int64)

        network = TabTransformerNetwork(
            self.n_features_in_, n_outputs,
            d_model=self.hidden_dim, n_heads=4, n_layers=self.n_layers, dropout=self.dropout,
        )
        config = DLConfig(
            lr=self.lr, weight_decay=self.weight_decay, epochs=self.epochs,
            batch_size=self.batch_size, patience=self.patience, random_state=self.random_state,
        )
        self.model_ = DLTrainer(config, self.task_type).train(network, X, y_train)
        return self
