"""
Configurable projection heads for retrieval: linear or MLP.
"""
import torch
import torch.nn as nn


def build_projection_head(
    in_dim: int,
    out_dim: int,
    head_type: str = "linear",
    hidden_dim: int = 512,
    num_layers: int = 2,
    dropout: float = 0.0,
):
    """
    head_type: "linear" -> single linear layer
               "mlp" -> Linear -> ReLU -> [Linear -> ReLU] * (num_layers-1) -> Linear
    """
    if head_type == "linear":
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
        )
    if head_type == "mlp":
        layers = []
        d = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        return nn.Sequential(*layers)
    raise ValueError(f"Unknown head_type: {head_type}")


class ProjectionHead(nn.Module):
    """Wrapper that optionally L2-normalizes output (for cosine similarity)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        head_type: str = "linear",
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.0,
        normalize: bool = True,
    ):
        super().__init__()
        self.proj = build_projection_head(
            in_dim, out_dim, head_type, hidden_dim, num_layers, dropout
        )
        self.normalize = normalize
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)
        if self.normalize:
            out = nn.functional.normalize(out, p=2, dim=-1)
        return out
