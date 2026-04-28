import torch
from torch import nn
import torch.nn.functional as F


def warp_features(F1, flow_patch):
        """
            F1: [B, C, H_patch, W_patch]
            flow_patch : [B, 2, H_patch, W_patch]

            returns:
            F1_warped : [B, D, H_patch, W_patch]
        """

        B, D, H, W = F1.shape

        y, x = torch.meshgrid(
            torch.arange(H, device=F1.device),
            torch.arange(W, device=F1.device),
            indexing='ij'
        )

        base_grid = torch.stack((x, y), dim=0).float() # [2, H, W]
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1) # [B, 2, H, W]

        sampling_grid = base_grid + flow_patch

        sampling_grid[:, 0, :, :] = 2.0 * sampling_grid[:, 0, :, :] / (W - 1) - 1.0
        sampling_grid[:, 1, :, :] = 2.0 * sampling_grid[:, 1, :, :] / (H - 1) - 1.0

        sampling_grid = sampling_grid.permute(0, 2, 3, 1)      # [B, H, W, 2]

        F1_warped = F.grid_sample(F1, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return F1_warped
