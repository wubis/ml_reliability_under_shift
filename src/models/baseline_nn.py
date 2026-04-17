from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class FeedforwardNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class NeuralNetClassifier:
    """
    sklearn-like wrapper around a PyTorch binary classifier.

    Common interface:
        - fit(X, y)
        - predict_proba(X)
        - predict(X)
        - predict_logits(X)
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 100,
        device: str | None = None,
        random_state: int = 42,
    ):
        self.hidden_dims = hidden_dims or [32, 16]
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.loss_history = []

    def _set_seed(self) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetClassifier":
        self._set_seed()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        input_dim = X.shape[1]
        self.model = FeedforwardNet(input_dim, self.hidden_dims).to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.loss_history = []

        self.model.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0

            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            self.loss_history.append(epoch_loss / len(loader))

        return self

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been fit yet.")

        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor).cpu().numpy().reshape(-1)

        return logits

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(X)

        probs_pos = torch.sigmoid(torch.tensor(logits)).numpy()
        probs_neg = 1.0 - probs_pos

        return np.column_stack([probs_neg, probs_pos])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs_pos = self.predict_proba(X)[:, 1]
        return (probs_pos >= threshold).astype(int)

    def get_params(self) -> dict:
        return {
            "hidden_dims": self.hidden_dims,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "device": self.device,
            "random_state": self.random_state,
        }

    def clone(self) -> "NeuralNetClassifier":
        return copy.deepcopy(self)