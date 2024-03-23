import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=1.2, reduction='mean'):
        """
        Focal loss for multilabel classification.
        Args:
            alpha (float, optional): Weighting factor for the rare class. Defaults to 0.25.
            gamma (float, optional): Focusing parameter to smooth the easy examples. Defaults to 2.0.
            reduction (str, optional): Specifies the reduction type: 'none' | 'mean' | 'sum'. Defaults to 'mean'.
        """
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute the focal loss given the model output (logits) and the ground truth labels.
        Args:
            inputs (torch.Tensor): Logits output by the model (before sigmoid).
            targets (torch.Tensor): Ground truth binary labels (same shape as inputs).

        Returns:
            torch.Tensor: Computed focal loss.
        """
        # Ensure the inputs and targets are the same size
        if inputs.size() != targets.size():
            raise ValueError(f"Target size ({targets.size()}) must be the same as input size ({inputs.size()})")

        # Calculate the binary cross-entropy loss without reduction
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Calculate probabilities
        probs = torch.sigmoid(inputs)
        # Calculate the modulating factor. For the positive class (targets == 1), this is (1 - p_t)**gamma.
        # For the negative class (targets == 0), this is (p_t)**gamma.
        modulating_factor = (1 - targets) * probs.pow(self.gamma) + targets * (1 - probs).pow(self.gamma)
        
        # Apply the alpha weighting
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # Compute the focal loss
        focal_loss = alpha_weight * modulating_factor * bce_loss

        # Apply the desired reduction
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError(f"Invalid reduction type '{self.reduction}'. Expected 'none', 'mean', or 'sum'.")