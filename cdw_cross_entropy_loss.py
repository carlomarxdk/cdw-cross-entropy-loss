import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CDW_CELoss(nn.Module):
    """
    Implementation of the Class Distance Weighted Cross-Entropy Loss 
    as described in https://arxiv.org/abs/2202.05167 (and further modified)
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 2.0,
        delta: float = 3.0,
        reduction: str = "mean",
        transform: str = "power",
        eps: float = 1e-8,
    ):
        super().__init__()
        assert alpha > 0, "Alpha should be larger than 0"
        assert reduction in ("none", "mean", "sum")
        assert transform in ("huber", "log", "power")

        self.reduction = reduction
        self.transform = transform
        self.alpha = alpha
        self.delta = delta
        self.eps = eps
        self.num_classes = num_classes
        self.register_buffer("w", torch.arange(num_classes, dtype=torch.float))

    def huber_transform(self, x: Tensor) -> Tensor:
        """Weight distances according to the Huber loss."""
        return torch.where(
            x < self.delta,
            0.5 * torch.pow(x, 2),
            self.delta * (x - 0.5 * self.delta),
        )

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            logits: (B, C) raw model outputs
            target: (B,) integer class labels
        Returns:
            loss: Tensor
        """
        probs = F.softmax(logits, dim=-1)
        
        weights = torch.abs(self.w - target.view(-1, 1))

        if self.transform == "huber":
            weights = self.huber_transform(weights)
        elif self.transform == "log":
            weights = torch.log1p(weights)
            weights = torch.pow(weights, self.alpha)
        elif self.transform == "power":
            weights = torch.pow(weights, self.alpha)

        # Weighted negative log-likelihood across all classes
        # Penalizes probability mass on distant classes
        loss = -torch.sum(weights * torch.log(1 - probs + self.eps), dim=-1)

        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        else:  # sum
            return loss.sum()
