import torch
import timm
import random
from torch import nn

from modules.masking import mask_tokens
from typing import Tuple, Union
from timm.models.vision_transformer import VisionTransformer
    
###############################################################################
# Encoder for EMA-Supervision in the decoder feature space
###############################################################################

class VisionTransformerWithPretrainedWts_decoderbased(VisionTransformer):
    def __init__(self, patch_size, img_size, **kwargs):
        """
        pretrained_cfg: pass the same kwargs you’d pass to timm.create_model

        """
        super().__init__(img_size=img_size,**kwargs)
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=patch_size, stride=patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))

    def forward_features_student(self, x, mask_ratio):
        x = self.patch_embed.proj(x)                    
        x = x.flatten(2).transpose(1, 2)   # [B, N, D]

        x_masked, masks, masked_indices = mask_tokens(x, mask_ratio, self.mask_token) 
        x = x_masked + self.pos_embed.to(x.device)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x
    
    def forward_features_teacher(self, x):
        x = self.patch_embed.proj(x)                    
        x = x.flatten(2).transpose(1, 2)   # [B, N, D]

        x = x + self.pos_embed.to(x.device)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x

class Encoder_decoderbased(nn.Module):
    def __init__(
        self, 
        resolution: Union[Tuple[int, int], int], 
        channels: int = 3, 
        pretrained_encoder = 'MAE',
        patch_size: int = 16,
        z_channels: int = 768,
        e_dim: int = 8,
        normalize_embedding: bool = True,
        # **ignore_kwargs
    ) -> None:
        # Initialize parent class with the first patch size
        super().__init__()
        self.image_size = resolution
        self.patch_size = patch_size
        self.channels = channels
        self.normalize_embedding = normalize_embedding
        self.z_channels = z_channels
        self.e_dim = e_dim
        
        self.init_transformer(pretrained_encoder)

    def init_transformer(self, pretrained_encoder):
        if pretrained_encoder == 'VIT_DINO':
            pretrained_encoder_model = 'timm/vit_base_patch16_224.dino'
        elif pretrained_encoder == 'VIT_DINOv2':
            pretrained_encoder_model = 'timm/vit_base_patch14_dinov2.lvd142m'
        elif pretrained_encoder == 'MAE':
            pretrained_encoder_model = 'timm/vit_base_patch16_224.mae'
        elif pretrained_encoder == 'MAE_VIT_L':
            pretrained_encoder_model = 'timm/vit_large_patch16_224.mae'
        elif pretrained_encoder == 'VIT':
            pretrained_encoder_model = 'timm/vit_large_patch32_224.orig_in21k'
        elif pretrained_encoder == 'CLIP32':
            pretrained_encoder_model = 'timm/vit_base_patch32_clip_224.openai'
        elif pretrained_encoder == 'CLIP':
            pretrained_encoder_model = 'timm/vit_base_patch16_clip_224.openai'
        elif pretrained_encoder == 'base':
            pretrained_encoder_model = 'timm/vit_base_patch16_224'
        elif pretrained_encoder == 'large':
            pretrained_encoder_model = 'timm/vit_large_patch16_224'
       

        # self.encoder = timm.create_model(pretrained_encoder_model, img_size=self.image_size, patch_size=self.patch_size, pretrained=False, dynamic_img_size=True).train()
        pretrained_model = timm.create_model(pretrained_encoder_model, img_size=self.image_size, patch_size=self.patch_size, pretrained=True)
        state_dict = pretrained_model.state_dict()
        self.encoder = VisionTransformerWithPretrainedWts_decoderbased(patch_size=self.patch_size, img_size=self.image_size) 
        state_dict['pos_embed'] = state_dict['pos_embed'][:, 1:, :]

        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)

        print(f"Loaded with {len(missing)} missing keys and {len(unexpected)} unexpected keys")
        print("Missing keys:")
        print(missing)
        print("Unexpected Keys: ")
        print(unexpected)
    
    def forward(self, img: torch.FloatTensor, mask_ratio, mode) -> torch.FloatTensor:
        if mode == "student":
            h = self.encoder.forward_features_student(img, mask_ratio)
        else:
            h = self.encoder.forward_features_teacher(img)
        h = h.permute(0, 2, 1).contiguous()
        h = h.reshape(h.shape[0], -1, img.size(2)//self.patch_size, img.size(3)//self.patch_size)
        return h