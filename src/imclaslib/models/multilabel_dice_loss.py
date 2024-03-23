import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid activation to predict probabilities
        inputs = torch.sigmoid(inputs)
        
        # Calculate intersection and union
        intersection = (inputs * targets).sum(dim=1)
        union = inputs.sum(dim=1) + targets.sum(dim=1)
        
        # Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss
        dice_loss = 1 - dice
        
        # Average Dice loss over the batch
        return dice_loss.mean()