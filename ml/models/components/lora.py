"""
LoRA (Low-Rank Adaptation) for linear layers.
W' = W + (alpha/r) * B @ A; W frozen, A and B trainable.
Used to adapt ViT/DINOv2 backbones with minimal trainable parameters.
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class LoRALayer(nn.Module):
    """
    Wraps nn.Linear with LoRA: output = linear(x) + (alpha / r) * (lora_B @ lora_A) @ x.
    Original linear is frozen; only lora_A (r, in_features) and lora_B (out_features, r) are trainable.
    """

    def __init__(
        self,
        original: nn.Linear,
        r: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original
        self.original.requires_grad_(False)
        in_features = original.in_features
        out_features = original.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self._init_lora()

    def _init_lora(self):
        # LoRA paper: A ~ N(0, sigma^2), B zero so initial delta = 0
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.original(x)
        # (B, *, in) @ A.T -> (B, *, r); then @ B.T -> (B, *, out)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return out + self.dropout(lora_out) * self.scaling


def _name_ends_with_any(name: str, patterns: List[str]) -> bool:
    """True if name ends with any of the patterns (e.g. 'blocks.0.attn.qkv' ends with 'attn.qkv')."""
    for p in patterns:
        if name.endswith(p) or name == p:
            return True
    return False


def inject_lora(
    module: nn.Module,
    target_module_patterns: List[str],
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> Tuple[int, List[str]]:
    """
    Replace nn.Linear submodules matching target_module_patterns with LoRALayer.
    target_module_patterns: e.g. ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"] so that
    backbone.blocks.0.attn.qkv matches "attn.qkv".
    Returns (num_injected, list of replaced module names).
    """
    if r <= 0:
        return 0, []
    injected = []
    # We need to replace in place; iterate by parent and child name
    for parent_name, parent in list(module.named_modules()):
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if not isinstance(child, nn.Linear):
                continue
            if not _name_ends_with_any(full_name, target_module_patterns):
                continue
            in_f = child.in_features
            out_f = child.out_features
            if in_f <= 0 or out_f <= 0:
                continue
            r_use = min(r, in_f, out_f)
            lora_layer = LoRALayer(child, r=r_use, alpha=alpha, dropout=dropout)
            setattr(parent, child_name, lora_layer)
            injected.append(full_name)
    return len(injected), injected


def get_trainable_lora_params(module: nn.Module) -> List[Tuple[str, int]]:
    """Return list of (name, numel) for trainable parameters (for reporting)."""
    out = []
    for name, p in module.named_parameters():
        if p.requires_grad:
            out.append((name, p.numel()))
    return out


def count_parameters(module: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable
