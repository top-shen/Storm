import torch
import torch.nn as nn


class StormTransformer(nn.Module):
    def __init__(
        self,
        num_assets: int,
        feature_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        max_len: int = 256,
    ):
        super().__init__()

        self.num_assets = num_assets
        self.feature_dim = feature_dim
        self.input_size = num_assets * feature_dim

        self.input_proj = nn.Linear(self.input_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_assets)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 4:
            raise ValueError(f"Expected features with shape (B, W, N, D), got {tuple(features.shape)}")

        batch_size, history_timestamps, num_assets, feature_dim = features.shape
        if num_assets != self.num_assets or feature_dim != self.feature_dim:
            raise ValueError(
                f"Expected input assets/features ({self.num_assets}, {self.feature_dim}), got ({num_assets}, {feature_dim})"
            )
        if history_timestamps > self.pos_embed.shape[1]:
            raise ValueError(
                f"Input history length {history_timestamps} exceeds max_len {self.pos_embed.shape[1]}"
            )

        x = features.reshape(batch_size, history_timestamps, num_assets * feature_dim)
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :history_timestamps, :]
        x = self.encoder(x)
        x = self.norm(x[:, -1, :])
        pred = self.head(x)
        return pred
