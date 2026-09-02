"""Multivariate LSTM forecaster over a spacecraft's telemetry channels."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from config import DROPOUT, HIDDEN, LAYERS


class TelemetryForecaster(nn.Module):
    """Predicts the next value of *every* channel from a window of history.

    Modelling all channels jointly (rather than one model per channel, as the
    NASA telemanom baseline does) is what lets the explainability layer compare
    per-channel prediction errors on a common footing and attribute an anomaly
    across subsystems.
    """

    def __init__(self, n_channels: int, n_cmd: int = 24,
                 hidden: int = HIDDEN, layers: int = LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.n_channels = n_channels
        self.lstm = nn.LSTM(
            input_size=n_channels + n_cmd,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, n_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, W, C+24) -> (B, C)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class ChannelScaler:
    """Per-channel standardisation fitted on training data only."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "ChannelScaler":
        self.mean = values.mean(axis=0)
        std = values.std(axis=0)
        # Constant channels would otherwise blow up; leave them untouched.
        self.std = np.where(std < 1e-6, 1.0, std)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (values * self.std + self.mean).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state: dict) -> "ChannelScaler":
        self.mean, self.std = state["mean"], state["std"]
        return self
