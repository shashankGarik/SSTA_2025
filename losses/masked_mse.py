import torch
import torch.nn.functional as F

def masked_mse(output, gt):
    gt_mask = (gt[:,0]!=1).float()
    bce = F.binary_cross_entropy_with_logits(output[:,0],gt_mask)

    mask = gt_mask.bool()

    if mask.any():
        masked_t2no_pred, masked_t2nd_pred  = output[:,1][mask], output[:,2][mask]
        masked_t2no_gt, masked_t2nd_gt      = gt[:,0][mask], gt[:,1][mask]

        mse_t2no = F.mse_loss(masked_t2no_pred, masked_t2no_gt)
        mse_t2nd = F.mse_loss(masked_t2nd_pred, masked_t2nd_gt)
    else:
        mse_t2no = torch.tensor(0.0, device=output.device)
        mse_t2nd = torch.tensor(0.0, device=output.device)

    total_loss = (1/10)*bce + mse_t2no + mse_t2nd

    return total_loss, bce, mse_t2no, mse_t2nd, mask.unsqueeze(1)