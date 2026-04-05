"""
Metric learning losses: Supervised Contrastive (SupCon), Triplet, InfoNCE / NT-Xent.
Expect inputs to be L2-normalized when using cosine similarity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised Contrastive Loss (Khosla et al.).
    features: (B, D) L2-normalized
    labels: (B,) class/item indices or hashable labels; same label = positive pair.
    """
    device = features.device
    B = features.size(0)
    if B < 2:
        return (features * 0.0).sum()

    features = F.normalize(features, p=2, dim=1)

    sim = torch.mm(features, features.t())  # (B, B), 값 범위 [-1, 1]
  
    max_logit = 50.0  # exp(50) ≈ 5e21 → float32 safe
    sim = (sim / temperature).clamp(min=-max_logit, max=max_logit)

    mask_self = torch.eye(B, dtype=torch.bool, device=device)
    sim = sim.masked_fill(mask_self, float("-inf"))

    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.t()) & ~mask_self  # (B, B) bool

    pos_per_anchor = positive_mask.sum(dim=1)  # (B,)
    valid = pos_per_anchor > 0
    if valid.sum() == 0:
        return (features * 0.0).sum()

    exp_sim = torch.exp(sim)
    exp_sim = exp_sim.masked_fill(mask_self, 0.0)
    sum_exp = exp_sim.sum(dim=1, keepdim=True)

    log_prob = sim - torch.log(sum_exp.clamp(min=1e-9))  # (B, B)

    log_prob = log_prob.masked_fill(mask_self, 0.0)

    loss_per_anchor = -(log_prob * positive_mask.float()).sum(dim=1) \
                      / pos_per_anchor.float().clamp(min=1.0)        # (B,)

    loss = loss_per_anchor[valid].mean()

    if not torch.isfinite(loss):
        return (features * 0.0).sum()

    return loss


def triplet_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
    hard_mining: bool = True,
) -> torch.Tensor:
    """
    Triplet loss: anchor, positive (same label), negative (different label).
    features: (B, D) L2-normalized -> distance = 2 - 2*cos = 2 - 2*dot
    """
    B = features.size(0)
    if B < 2:
        return (features * 0.0).sum()

    # Pairwise squared distance = 2 - 2 * cos (when normalized)
    dot = torch.mm(features, features.t())
    dist = (2.0 - 2.0 * dot).clamp(min=0.0)

    labels = labels.view(-1, 1)
    same = labels == labels.t()
    diff = ~same

    losses = []
    for i in range(B):
        pos_mask = same[i].clone()
        pos_mask[i] = False
        neg_mask = diff[i]
        if not pos_mask.any() or not neg_mask.any():
            continue
        d_ap = dist[i][pos_mask]
        d_an = dist[i][neg_mask]
        if hard_mining:
            d_ap = d_ap.min()
            d_an = d_an.min()
        else:
            d_ap = d_ap.mean()
            d_an = d_an.mean()
        loss = F.relu(d_ap - d_an + margin)
        losses.append(loss)
    if not losses:
        return (features * 0.0).sum()
    return torch.stack(losses).mean()


def infonce_nt_xent_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE / NT-Xent: for each anchor, positives are same-label samples (excluding self).
    Cross-entropy of positive among all (except self).
    features: (B, D) L2-normalized
    """
    device = features.device
    B = features.size(0)
    if B < 2:
        return (features * 0.0).sum()

    sim = torch.mm(features, features.t()) / temperature
    mask_self = torch.eye(B, dtype=torch.bool, device=device)
    sim = sim.masked_fill(mask_self, float("-inf"))

    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.t()) & ~mask_self
    pos_per_anchor = positive_mask.sum(dim=1)
    valid = pos_per_anchor > 0
    if valid.sum() == 0:
        return (features * 0.0).sum()

    logits = sim
    # Target: for each row, positive indices (multi-label); we use mean CE over positives
    log_probs = F.log_softmax(logits, dim=1)
    log_probs = log_probs.masked_fill(mask_self, 0.0)
    # For each anchor i: loss = - mean over j in positives(i) of log_probs[i, j]
    loss_per_anchor = -(log_probs * positive_mask.float()).sum(dim=1) / (pos_per_anchor.float() + 1e-10)
    return loss_per_anchor[valid].mean()


def get_metric_loss(
    loss_type: str,
    temperature: float = 0.07,
    margin: float = 0.2,
    hard_mining: bool = True,
):
    """Return a callable loss(embeddings, labels) -> scalar."""
    if loss_type.lower() in ("supcon", "supervised_contrastive"):
        return lambda feats, labels: supervised_contrastive_loss(feats, labels, temperature=temperature)
    if loss_type.lower() in ("triplet", "triplet_loss"):
        return lambda feats, labels: triplet_loss(feats, labels, margin=margin, hard_mining=hard_mining)
    if loss_type.lower() in ("infonce", "nt_xent", "ntxent"):
        return lambda feats, labels: infonce_nt_xent_loss(feats, labels, temperature=temperature)
    raise ValueError(f"Unknown loss_type: {loss_type}")
