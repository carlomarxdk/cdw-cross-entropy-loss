import torch
import torch.nn as nn
from torch import Tensor


class CDW_CELoss(nn.Module):
    """
    Implementation of the Class Distance Weighted Cross-Entropy Loss as described in https://arxiv.org/abs/2202.05167"""

    def __init__(self, num_classes: int,
                 alpha: float = 2.,
                 delta: float = 3.,
                 reduction: str = "mean",
                 transform: str = "power",  # Original paper uses power transform
                 eps: float = 1e-8):
        super(CDW_CELoss, self).__init__()
        """
        :param num_classes: Name of the vocabulary.
        :param alpha: Exponent for the penalty of the distance, 
        :param delta: Threshold for the Huber transform,
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        :param transform: Specifies the transformation to apply to the distance/penalty: 'huber' | 'log' | 'power'
        """

        assert alpha > 0, "Alpha should be larger than 0"
        assert reduction in [
            "mean", "sum"], "Reduction should be either mean or sum"
        assert transform in [
            "huber", "log", "power"], "Transform should be either huber, log or power"

        self.reduction = reduction
        self.transform = transform
        self.alpha = alpha
        self.eps = eps
        self.num_classes = num_classes
        self.register_buffer(name="w", tensor=torch.tensor(
            [float(i) for i in range(self.num_classes)]))  # to speed up the computation

        self.delta = delta  # for huber transform only

    def huber_transform(self, x):
        """Weight distances according to the Huber Loss"""
        return torch.where(
            x < self.delta,
            0.5 * torch.pow(x, 2),
            self.delta * (x - 0.5 * self.delta)
        )

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        Forward Methods
        :param logits: Logits from the model (Torch.Tensor) Size: (B, C)
        :param target: Target labels (Torch.Tensor) Size: (B, )

        :return: Loss value (Torch.Tensor)
        """
        w = torch.abs(self.w - target.view(-1, 1))  # calculate penalty weights

        if self.transform == "huber":
            # apply huber transform (not in the paper)
            w = self.huber_transform(w)
        elif self.transform == "log":
            w = torch.log1p(w)
            # apply log transform (not in the paper)
            w = torch.pow(w, self.alpha)
        elif self.transform == "power":
            # apply power transform (in the paper)
            w = torch.pow(w, self.alpha)
        else:
            raise NotImplementedError(
                "%s transform is not implemented" % self.transform)

        loss = - torch.mul(torch.log(1 - logits + self.eps), w).sum(-1)

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            raise NotImplementedError(
                "%s reduction is not implemented" % self.reduction)
