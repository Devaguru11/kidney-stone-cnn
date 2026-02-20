# src/training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy examples so model focuses on hard ones.
    gamma=2.0: higher = more focus on hard examples
    alpha=0.75: weight for positive class (stone)
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t     = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal   = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        return focal.mean()

