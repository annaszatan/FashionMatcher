"""
Backbone wrapper for DINOv2 ViT: load from hub, freeze/unfreeze, optional projection head.
Supports: frozen, linear_probe (freeze backbone, train head), partial_finetune (last N blocks + head),
full_finetune. Optional LoRA is supported via a simple adapter (optional dependency).
"""
import os
import sys
from typing import Optional

import torch
import torch.nn as nn

# Ensure project root and dinov2 are on path when running from project root
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_DINOV2_ROOT = os.path.join(_PROJECT_ROOT, "models", "dinov2")
if os.path.isdir(_DINOV2_ROOT) and _DINOV2_ROOT not in sys.path:
    sys.path.insert(0, _DINOV2_ROOT)

from models.projection_head import ProjectionHead
from models.components.lora import inject_lora

def load_dino_backbone(backbone_name: str, device: str = "cuda"):
    """Load DINOv2 backbone (ViT with Identity head) from dinov2.hub.backbones."""
    allowed = {"dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"}
    if backbone_name not in allowed:
        raise ValueError(f"Unknown DINOv2 backbone: {backbone_name}")

    # torch.hub downloads the model once and uses the local cache afterward.
    model = torch.hub.load("facebookresearch/dinov2", backbone_name, pretrained=True, verbose=False)

    model.eval()
    model.to(device)
    return model


class RetrievalBackbone(nn.Module):
    """
    Wrapper: DINOv2 backbone + optional projection head.
    - backbone: ViT from dinov2, outputs embed_dim (e.g. 768 for ViT-B)
    - head: Identity (use backbone dim as-is) or ProjectionHead
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vitb14",
        embedding_dim: int = 256,
        head_type: str = "linear",
        head_hidden_dim: int = 512,
        head_num_layers: int = 2,
        head_dropout: float = 0.0,
        normalize_output: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.backbone = load_dino_backbone(backbone_name, device)
        self.backbone_name = backbone_name
        embed_dim = getattr(self.backbone, "embed_dim", self.backbone.num_features)

        self.normalize_output = normalize_output
        if head_type is None or head_type == "identity":
            self.head = nn.Identity()
            self.embed_dim = embed_dim
            self.out_dim = embed_dim
        else:
            self.head = ProjectionHead(
                in_dim=embed_dim,
                out_dim=embedding_dim,
                head_type=head_type,
                hidden_dim=head_hidden_dim,
                num_layers=head_num_layers,
                dropout=head_dropout,
                normalize=normalize_output,
            )
            self.embed_dim = embed_dim
            self.out_dim = embedding_dim
        self._device = device

    def forward(self, x: torch.Tensor, is_training: bool = False) -> torch.Tensor:
        feat = self.backbone.forward_features(x)
        if isinstance(feat, dict):
            feat = feat["x_norm_clstoken"]
        out = self.head(feat)
        if self.normalize_output and isinstance(self.head, nn.Identity):
            out = nn.functional.normalize(out, p=2, dim=-1)
        return out

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def unfreeze_last_n_blocks(self, n: int):
        """Unfreeze the last n transformer blocks; keep rest frozen."""
        self.freeze_backbone()
        blocks = self.backbone.blocks
        if hasattr(blocks, "__len__"):
            total = len(blocks)
            if getattr(self.backbone, "chunked_blocks", False):
                # BlockChunk: each element is a chunk (ModuleList of blocks)
                start_idx = max(0, total - n)
                for i in range(start_idx, total):
                    chunk = blocks[i]
                    for p in chunk.parameters():
                        p.requires_grad = True
            else:
                start_idx = max(0, total - n)
                for i in range(start_idx, total):
                    for p in blocks[i].parameters():
                        p.requires_grad = True

    def train_head_only(self):
        self.freeze_backbone()
        for p in self.head.parameters():
            p.requires_grad = True

    def train_partial_finetune(self, num_unfrozen_blocks: int = 2):
        self.unfreeze_last_n_blocks(num_unfrozen_blocks)
        for p in self.head.parameters():
            p.requires_grad = True

    def train_full_finetune(self):
        self.unfreeze_backbone()
        for p in self.head.parameters():
            p.requires_grad = True

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def apply_lora(
        self,
        target_module_patterns: list,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> tuple:
        """
        Inject LoRA into backbone Linear layers matching target_module_patterns.
        Returns (num_injected, list of injected module names).
        """
        n, names = inject_lora(
            self.backbone,
            target_module_patterns=target_module_patterns,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )
        return n, names

    def train_lora_peft(self, train_projection_head: bool = True):
        """
        Freeze all backbone parameters except LoRA (lora_A, lora_B).
        Optionally train projection head.
        """
        for name, p in self.backbone.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
        if train_projection_head:
            for p in self.head.parameters():
                p.requires_grad = True
        else:
            for p in self.head.parameters():
                p.requires_grad = False

    def get_trainable_param_report(self) -> dict:
        """Return total params, trainable params, and list of trainable module names (for metadata)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        trainable_names = [name for name, p in self.named_parameters() if p.requires_grad]
        return {
            "total_params": total,
            "trainable_params": trainable,
            "percentage_trainable": (100.0 * trainable / total) if total > 0 else 0.0,
            "trainable_param_names": trainable_names,
        }


def build_retrieval_model(
    backbone_name: str = "dinov2_vitb14",
    training_strategy: str = "frozen_baseline",
    embedding_dim: int = 256,
    head_type: str = "linear",
    head_hidden_dim: int = 512,
    head_num_layers: int = 2,
    num_unfrozen_blocks: int = 2,
    normalize_output: bool = True,
    device: str = "cuda",
    # LoRA-specific (used when training_strategy == "lora_peft")
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    lora_target_modules: Optional[list] = None,
    train_projection_head: bool = True,
) -> RetrievalBackbone:
    """
    training_strategy: frozen_baseline | linear_probe | partial_finetune | lora_peft | full_finetune
    """
    if training_strategy == "frozen_baseline":
        head_type = "identity"
    model = RetrievalBackbone(
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        head_type=head_type,
        head_hidden_dim=head_hidden_dim,
        head_num_layers=head_num_layers,
        normalize_output=normalize_output,
        device=device,
    )
    if training_strategy == "frozen_baseline":
        model.freeze_backbone()
        if not isinstance(model.head, nn.Identity):
            for p in model.head.parameters():
                p.requires_grad = False
    elif training_strategy == "linear_probe":
        model.train_head_only()
    elif training_strategy == "partial_finetune":
        model.train_partial_finetune(num_unfrozen_blocks=num_unfrozen_blocks)
    elif training_strategy == "lora_peft":
        # Real LoRA: inject adapters, freeze backbone, train only LoRA + optional head
        if lora_target_modules is None:
            lora_target_modules = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
        num_injected, injected_names = model.apply_lora(
            target_module_patterns=lora_target_modules,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        if num_injected == 0:
            raise RuntimeError(
                "LoRA injection found no target modules. Check lora_target_modules "
                "match the backbone (e.g. attn.qkv, attn.proj, mlp.fc1, mlp.fc2 for ViT)."
            )
        model.train_lora_peft(train_projection_head=train_projection_head)
    elif training_strategy == "full_finetune":
        model.train_full_finetune()
    else:
        raise ValueError(f"Unknown training_strategy: {training_strategy}")
    return model
