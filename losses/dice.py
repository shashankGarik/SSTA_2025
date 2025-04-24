import torch

def dice_loss(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)  # Ensure preds are probabilities
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    intersection = (preds_flat * targets_flat).sum()
    return 1 - (2. * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)