import torch
import timm
import random
from torch import nn

from modules.masking import mask_tokens
from typing import Tuple, Union
from timm.models.vision_transformer import VisionTransformer

class VisionTransformerWithPretrainedWts(VisionTransformer):
    def __init__(self, patch_size, img_size, mask_ratio, **kwargs):
        """
        pretrained_cfg: pass the same kwargs you’d pass to timm.create_model

        """
        super().__init__(img_size=img_size,**kwargs)
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio

        self.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=patch_size, stride=patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))

    def forward_features(self, x, mask_ratio):
        x = self.patch_embed.proj(x)                    
        x = x.flatten(2).transpose(1, 2)   # [B, N, D]

        x_masked, masks, masked_indices = mask_tokens(x, mask_ratio, self.mask_token) 
        # print("Shape of masked version of img : ", x_masked.shape)
        x = x_masked + self.pos_embed.to(x.device)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x

class Encoder(nn.Module):
    def __init__(
        self, 
        mask_ratio,
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
        self.mask_ratio = mask_ratio
        
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
        self.encoder = VisionTransformerWithPretrainedWts(patch_size=self.patch_size, img_size=self.image_size, 
                                                          mask_ratio=self.mask_ratio) 
        state_dict['pos_embed'] = state_dict['pos_embed'][:, 1:, :]

        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)

        print(f"Loaded with {len(missing)} missing keys and {len(unexpected)} unexpected keys")
        print("Missing keys:")
        print(missing)
        print("Unexpected Keys: ")
        print(unexpected)
    
    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        h = self.encoder.forward_features(img, self.mask_ratio)
        h = h.permute(0, 2, 1).contiguous()
        h = h.reshape(h.shape[0], -1, img.size(2)//self.patch_size, img.size(3)//self.patch_size)
        return h

class VisionTransformerWithPretrainedWtsMultiframes_1dtokens(VisionTransformer):
    def __init__(self, patch_size, img_size, num_extra_tokens, **kwargs):
        """
        pretrained_cfg: pass the same kwargs you’d pass to timm.create_model

        """
        super().__init__(img_size=img_size,**kwargs)
        self.num_patches = (img_size // patch_size) ** 2
        self.num_extra_tokens = num_extra_tokens

        self.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=patch_size, stride=patch_size)
        self.prompt_tokens = nn.Parameter(torch.zeros(1, self.num_extra_tokens, self.embed_dim))
        new_length = self.num_patches + self.num_extra_tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, new_length, self.embed_dim))

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed.proj(x) 
        x = x.flatten(2).transpose(1, 2)   # [B, N, D]
        print("Shape of x1: ", x.shape)

        if self.num_extra_tokens > 1:
            prompt_tokens = self.prompt_tokens.expand(B, -1, -1)
            x = torch.cat((x, prompt_tokens), dim=1)
            print("Shape of x2 : ", x.shape)

        x = x + self.pos_embed.to(x.device)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        latents_1d = x[:, self.num_patches:]

        return latents_1d

class Encoder_1dtokens(nn.Module):
    def __init__(
        self, 
        mask_ratio,
        num_extra_tokens,
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
        self.mask_ratio = mask_ratio
        self.num_extra_tokens = num_extra_tokens
        
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
        K = self.num_extra_tokens

        self.encoder = VisionTransformerWithPretrainedWtsMultiframes_1dtokens(patch_size=self.patch_size, img_size=self.image_size, num_extra_tokens=self.num_extra_tokens) 
        
        state_dict['pos_embed'] = nn.Parameter(torch.zeros(1, (state_dict['pos_embed'].shape[1]-1)+K, 768))

        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)

        print(f"Loaded with {len(missing)} missing keys and {len(unexpected)} unexpected keys")
        print("Missing keys:")
        print(missing)
        print("Unexpected Keys: ")
        print(unexpected)
    
    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
       
        h = self.encoder.forward_features(img) # [B, N, D]
        
        return h