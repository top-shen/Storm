import torch
import torch.nn as nn


class StormLSTM(nn.Module):
    def __init__(
        self,
        num_assets: int,
        feature_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_assets = num_assets
        self.feature_dim = feature_dim
        self.input_size = num_assets * feature_dim

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, num_assets)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 4:
            raise ValueError(f"Expected features with shape (B, W, N, D), got {tuple(features.shape)}")

        batch_size, history_timestamps, num_assets, feature_dim = features.shape
        if num_assets != self.num_assets or feature_dim != self.feature_dim:
            raise ValueError(
                f"Expected input assets/features ({self.num_assets}, {self.feature_dim}), "
                f"got ({num_assets}, {feature_dim})"
            )

        features = features.reshape(batch_size, history_timestamps, num_assets * feature_dim)
        outputs, _ = self.lstm(features)
        pred = self.head(outputs[:, -1, :])
        return pred
