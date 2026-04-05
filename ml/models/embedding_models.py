from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Pooling
# ──────────────────────────────────────────────────────────────────────────────

class GeMPool(nn.Module):
    def __init__(self, p_init: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p_init))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, D)"""
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=1).pow(1.0 / p)


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """Stochastic depth."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device, dtype=x.dtype).add_(keep).floor_()
        return x * mask / keep


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_val = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_out = nn.Linear(hidden_dim, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.drop(F.silu(self.w_gate(x)) * self.w_val(x)))


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight self-attention block (no positional encoding)
# ──────────────────────────────────────────────────────────────────────────────

class LightAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        ls_init: float = 1e-4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.ls1 = LayerScale(d_model, init_value=ls_init)
        self.drop1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ff_hidden_dim, dropout=dropout)
        self.ls2 = LayerScale(d_model, init_value=ls_init)
        self.drop2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(self.ls1(h))

        h = self.ffn(self.norm2(x))
        x = x + self.drop2(self.ls2(h))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class TransformerEmbeddingModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        token_dim: int,
        output_dim: int = 256,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_hidden_dim: int = 512,
        dropout: float = 0.1,
        drop_path: float = 0.05,
        ls_init: float = 1e-4,
        gem_p_init: float = 3.0,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.token_dim = token_dim
        self.use_cls_token = use_cls_token

        # ── input projection ──
        self.input_proj = nn.Linear(token_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # ── transformer blocks (1~2 layers, no positional encoding) ──
        dp_rates = torch.linspace(0, drop_path, num_layers).tolist()
        self.blocks = nn.ModuleList([
            LightAttentionBlock(
                d_model=d_model,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                dropout=dropout,
                drop_path=dp_rates[i],
                ls_init=ls_init,
            )
            for i in range(num_layers)
        ])

        # ── final norm ──
        self.final_norm = nn.LayerNorm(d_model)

        # ── pooling ──
        self.gem_pool = GeMPool(p_init=gem_p_init)

        # ── output projection ──
        proj_input_dim = d_model * 2 if use_cls_token else d_model
        self.output_proj = nn.Sequential(
            nn.Linear(proj_input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            bsz = x.shape[0]
            x = x.view(bsz, self.seq_len, self.token_dim)
        elif x.dim() != 3:
            raise ValueError(f"Expected (B, D) or (B, L, D), got shape {tuple(x.shape)}")

        if self.use_cls_token:
            cls_token = x[:, 0:1, :]       # (B, 1, token_dim)
            patch_tokens = x[:, 1:, :]     # (B, L-1, token_dim)
        else:
            patch_tokens = x

        # input projection
        patch_tokens = self.input_norm(self.input_proj(patch_tokens))

        # transformer blocks
        for blk in self.blocks:
            patch_tokens = blk(patch_tokens)

        patch_tokens = self.final_norm(patch_tokens)

        # pooling
        gem_out = self.gem_pool(patch_tokens)   # (B, d_model)

        if self.use_cls_token:
            cls_proj = self.input_proj(cls_token.squeeze(1))  # (B, d_model)
            combined = torch.cat([cls_proj, gem_out], dim=-1)  # (B, 2*d_model)
            out = self.output_proj(combined)
        else:
            out = self.output_proj(gem_out)

        return F.normalize(out, p=2, dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# CNN model (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

class CNNEmbeddingModel(nn.Module):
    """1D CNN over sequence."""

    def __init__(
        self,
        output_dim: int = 256,
        channels=(64, 128, 256),
        kernel_size: int = 3,
        dropout: float = 0.1,
        *,
        seq_len: Optional[int] = None,
        token_dim: Optional[int] = None,
        input_dim: Optional[int] = None,
    ):
        super().__init__()
        if seq_len is not None and token_dim is not None:
            self.patch_mode = True
            self.seq_len = seq_len
            self.token_dim = token_dim
            in_ch = token_dim
        elif input_dim is not None:
            self.patch_mode = False
            self.input_dim = input_dim
            in_ch = 1
        else:
            raise ValueError("CNNEmbeddingModel requires (seq_len, token_dim) or input_dim")

        layers = []
        for ch in channels:
            layers.extend([
                nn.Conv1d(in_ch, ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(ch),
                nn.GELU(),
            ])
            in_ch = ch
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_mode:
            if x.dim() != 3:
                raise ValueError(f"Patch mode expects (B, L, D), got {tuple(x.shape)}")
            x = x.transpose(1, 2)
        else:
            if x.dim() != 2:
                raise ValueError(f"Legacy mode expects (B, D), got {tuple(x.shape)}")
            x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────────────────────

def build_embedding_model(
    model_type: str,
    cfg: dict,
    *,
    input_dim: Optional[int] = None,
    seq_len: Optional[int] = None,
    token_dim: Optional[int] = None,
) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "transformer":
        if seq_len is not None and token_dim is not None:
            s, td = seq_len, token_dim
        elif input_dim is not None:
            s = int(cfg.get("seq_len", 16))
            if input_dim % s != 0:
                raise ValueError(
                    f"input_dim({input_dim}) must be divisible by seq_len({s}) for legacy flat input"
                )
            td = input_dim // s
        else:
            raise ValueError("Transformer needs input_dim or (seq_len, token_dim)")
        return TransformerEmbeddingModel(
            seq_len=s,
            token_dim=td,
            output_dim=int(cfg.get("output_dim", 256)),
            d_model=int(cfg.get("d_model", 256)),
            num_layers=int(cfg.get("num_layers", 2)),
            num_heads=int(cfg.get("num_heads", 8)),
            ff_hidden_dim=int(cfg.get("ff_hidden_dim", 512)),
            dropout=float(cfg.get("dropout", 0.1)),
            drop_path=float(cfg.get("drop_path", 0.05)),
            ls_init=float(cfg.get("ls_init", 1e-4)),
            gem_p_init=float(cfg.get("gem_p_init", 3.0)),
            use_cls_token=bool(cfg.get("use_cls_token", True)),
        )
    if model_type == "cnn":
        channels = cfg.get("channels", [64, 128, 256])
        if seq_len is not None and token_dim is not None:
            return CNNEmbeddingModel(
                output_dim=int(cfg.get("output_dim", 256)),
                channels=tuple(int(c) for c in channels),
                kernel_size=int(cfg.get("kernel_size", 3)),
                dropout=float(cfg.get("dropout", 0.1)),
                seq_len=seq_len,
                token_dim=token_dim,
            )
        if input_dim is not None:
            return CNNEmbeddingModel(
                output_dim=int(cfg.get("output_dim", 256)),
                channels=tuple(int(c) for c in channels),
                kernel_size=int(cfg.get("kernel_size", 3)),
                dropout=float(cfg.get("dropout", 0.1)),
                input_dim=input_dim,
            )
        raise ValueError("CNN needs input_dim or (seq_len, token_dim)")
    raise ValueError(f"Unknown embedding model_type: {model_type}")
