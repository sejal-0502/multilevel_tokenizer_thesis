import torch
import torch.nn as nn

def mae_random_masking(x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        B, N, D = x.shape 
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]
        
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


def mask_tokens(x, mask_ratio, mask_token):
    """
    Apply learned mask token to a sequence of embeddings.

    Args:
        x: Tensor [B, N, D] - input token embeddings
        mask_ratio: float - fraction of tokens to mask
        mask_token: nn.Parameter of shape [1, 1, D] - learned mask token

    Returns:
        x_masked: Tensor [B, N, D] - input with masked tokens replaced
        mask: Tensor [B, N] float - 1 for masked, 0 for visible
        mask_indices: Tensor [B, num_masks] - indices of masked tokens
    """
    B, N, D = x.shape
    num_masks = int(N * mask_ratio)

    # generate random permutation of indices for masking
    rand_idx = torch.rand(B, N, device=x.device).argsort(dim=1)
    mask_indices = rand_idx[:, :num_masks]  # first num_masks indices to mask

    # create mask tensor [B, N], 1=masked, 0=visible
    mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
    batch_idx = torch.arange(B, device=x.device).unsqueeze(-1).expand(-1, num_masks)
    mask[batch_idx, mask_indices] = 1

    # replace masked positions with learned mask token
    mask_token_expanded = mask_token.expand(B, N, D)  # broadcast
    x_masked = x * (~mask.unsqueeze(-1)) + mask_token_expanded * mask.unsqueeze(-1)

    return x_masked, mask.float(), mask_indices

def mask_tokens_discard(x, mask_ratio):
    """
    Discard mask token from a sequence of embeddings.

    Args:
        x: Tensor [B, N, D] - input token embeddings
        mask_ratio: float - fraction of tokens to mask

    Returns:
        x_visible: Tensor [B, N_visible, D] - input with visible tokens only
        mask: Tensor [B, N] float - 1 for masked, 0 for visible
        mask_indices: Tensor [B, num_masks] - indices of masked tokens
    """
    B, N, D = x.shape
    num_masks = int(N * mask_ratio)
    num_visible = N - num_masks

    # generate random permutation of indices for masking
    rand_idx = torch.rand(B, N, device=x.device).argsort(dim=1)
    mask_indices = rand_idx[:, :num_masks]  # first num_masks indices to mask
    visible_indices = rand_idx[:, num_masks:]

    # create mask tensor [B, N], 1=masked, 0=visible
    mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
    batch_idx = torch.arange(B, device=x.device).unsqueeze(-1).expand(-1, num_masks)
    mask[batch_idx, mask_indices] = 1

    # select visible tokens only
    x_visible = torch.stack([x[b, visible_indices[b]] for b in range(B)], dim=0) # [B, N_visible, D]

    return x_visible, mask.float(), mask_indices



def tube_mask_two_frames(x1, x2, mask_ratio_frame1, mask_ratio_frame2, mask_token):
    """
    Tube/block masking for 2-frame MAE setup.
    
    Args:
        x1: [B, N, D] - frame 1 token embeddings
        x2: [B, N, D] - frame 2 token embeddings
        mask_ratio_frame1: float - fraction of tokens in frame1 to discard
        mask_ratio_frame2: float - fraction of tokens in frame2 to mask
        mask_token: nn.Parameter [1, 1, D] - learnable token for frame2 masked patches
    
    Returns:
        x1_visible: [B, N1_visible, D] - frame1 visible tokens
        x2_masked: [B, N2, D] - frame2 visible tokens with masked positions made learnable

    """
    B, N, D = x1.shape
    assert N == x2.shape[1], "Frame1 and Frame2 must have same token count"

    # -------------------------------
    # Step 1: Frame 1 masking (discarded)
    # -------------------------------
    num_mask1 = int(N * mask_ratio_frame1)
    rand_idx1 = torch.rand(B, N, device=x1.device).argsort(dim=1)
    mask_idx1 = rand_idx1[:, :num_mask1]
    visible_idx1 = rand_idx1[:, num_mask1:]

    mask_frame1 = torch.zeros(B, N, device=x1.device, dtype=torch.bool)
    batch_idx = torch.arange(B, device=x1.device).unsqueeze(-1).expand(-1, num_mask1)
    mask_frame1[batch_idx, mask_idx1] = 1

    x1_visible = torch.stack([x1[b, visible_idx1[b]] for b in range(B)], dim=0)

    # -------------------------------
    # Step 2: Frame 2 masking (tube + include frame1 masks)
    # -------------------------------
    # Start by marking all frame1 masked tokens as masked in frame2
    mask_frame2 = mask_frame1.clone()

    # Add additional random masks for frame2
    num_additional_mask2 = int(N * mask_ratio_frame2) - num_mask1
    if num_additional_mask2 > 0:
        rand_idx2 = torch.rand(B, N, device=x2.device).argsort(dim=1)
        # pick tokens that are NOT already masked
        add_mask_idx2 = []
        for b in range(B):
            avail = [i for i in rand_idx2[b].tolist() if not mask_frame2[b, i]]
            for idx in avail[:num_additional_mask2]:
                mask_frame2[b, idx] = 1

    # masks replaced by learnable tokens
    # x2: [B, N, D], mask_frame2: [B, N] bool, mask_token: [1, 1, D]
    mask_token_expanded = mask_token.expand(x2.size(0), x2.size(1), x2.size(2))
    x2_masked = x2 * (~mask_frame2.unsqueeze(-1)) + mask_token_expanded * mask_frame2.unsqueeze(-1)

    return x1_visible, x2_masked