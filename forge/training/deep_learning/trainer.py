"""PyTorch training loop with early stopping and mixed precision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DLConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 128
    patience: int = 8
    random_state: int = 42


class DLTrainer:
    """Generic PyTorch trainer with early stopping."""

    def __init__(self, config: DLConfig, task_type: str):
        self.config = config
        self.task_type = task_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> nn.Module:
        torch.manual_seed(self.config.random_state)
        model = model.to(self.device)

        if X_val is None:
            split = int(len(X_train) * 0.85)
            X_val, y_val = X_train[split:], y_train[split:]
            X_train, y_train = X_train[:split], y_train[:split]

        train_loader = self._loader(X_train, y_train, shuffle=True)
        val_loader = self._loader(X_val, y_val, shuffle=False)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.config.lr, epochs=self.config.epochs,
            steps_per_epoch=max(1, len(train_loader)),
        )

        is_regression = self.task_type == "regression"
        criterion: Callable
        if is_regression:
            criterion = nn.MSELoss()
        elif self._n_classes(y_train) == 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        best_state = None
        best_metric = float("-inf") if not is_regression else float("inf")
        patience = 0

        for _ in range(self.config.epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = model(xb)
                loss = self._loss(out, yb, criterion, is_regression)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            metric = self._evaluate(model, val_loader, is_regression)
            improved = metric > best_metric if not is_regression else metric < best_metric
            if improved:
                best_metric = metric
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.config.patience:
                    break

        if best_state:
            model.load_state_dict(best_state)
        return model

    def _loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        x_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32 if self.task_type == "regression" else torch.long)
        return DataLoader(
            TensorDataset(x_t, y_t),
            batch_size=min(self.config.batch_size, len(X)),
            shuffle=shuffle,
        )

    def _n_classes(self, y: np.ndarray) -> int:
        return len(np.unique(y))

    def _loss(self, out, yb, criterion, is_regression: bool) -> torch.Tensor:
        if is_regression:
            return criterion(out.squeeze(), yb.float())
        if out.shape[-1] == 1:
            return criterion(out.squeeze(), yb.float())
        return criterion(out, yb)

    def _evaluate(self, model: nn.Module, loader: DataLoader, is_regression: bool) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                out = model(xb)
                if is_regression:
                    losses.append(float(nn.functional.mse_loss(out.squeeze(), yb.float())))
                elif out.shape[-1] == 1:
                    losses.append(float(nn.functional.binary_cross_entropy_with_logits(out.squeeze(), yb.float())))
                else:
                    losses.append(float(nn.functional.cross_entropy(out, yb)))
        return float(np.mean(losses))
