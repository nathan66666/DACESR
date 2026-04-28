import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class KLDivLoss(nn.Module):
    """KL Divergence loss.

    Args:
        loss_weight (float): Loss weight for KL divergence loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none', 'batchmean', 'mean', 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(KLDivLoss, self).__init__()
        if reduction not in ['none', 'batchmean', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: none, batchmean, mean, sum')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        # Convert pred and target to probability distributions
        pred_prob = F.softmax(pred, dim=1)
        target_prob = F.softmax(target, dim=1)

        # Compute log probabilities
        pred_log_prob = torch.log(pred_prob + 1e-8)

        # Compute KL divergence
        loss = F.kl_div(pred_log_prob, target_prob, reduction=self.reduction)

        return self.loss_weight * loss
