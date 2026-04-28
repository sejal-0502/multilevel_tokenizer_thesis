import torch
import math
import timm
import random
from torch import nn
import torch.nn.functional as F

from modules.masking import mask_tokens, mask_tokens_discard, tube_mask_two_frames
from typing import Tuple, Union
from timm.models.vision_transformer import VisionTransformer
from modules.warping import warp_features

class VAEEncoder(nn.Module):
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
       

        self.encoder = timm.create_model(pretrained_encoder_model, img_size=self.image_size, patch_size=self.patch_size, pretrained=False, dynamic_img_size=True).train()
        pretrained_model = timm.create_model(pretrained_encoder_model, img_size=self.image_size, patch_size=self.patch_size, pretrained=True)
        """Initialize weights of target_model with weights from source_model."""
        with torch.no_grad():
            for target_param, source_param in zip(self.encoder.parameters(), pretrained_model.parameters()):
                target_param.data.copy_(source_param.data)

        del pretrained_model
    
    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        # print("Shape of img : ", img.shape)
        h = self.encoder.forward_features(img)[:,1:]
        h = h.permute(0, 2, 1).contiguous()
        h = h.reshape(h.shape[0], -1, img.size(2)//self.patch_size, img.size(3)//self.patch_size)
        return h

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
    """Encoder : Fixed Masking"""
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
    
class VisionTransformerWithPretrainedWts_mae(VisionTransformerWithPretrainedWts):
    def __init__(self, patch_size, img_size, mask_ratio, **kwargs):
        """
        1 Frame masking mae for weighted masking loss

        """
        super().__init__(patch_size, img_size, mask_ratio,**kwargs)

    def forward_features(self, x, mask_ratio):
        x = self.patch_embed.proj(x)                    
        x = x.flatten(2).transpose(1, 2)   # [B, N, D]

        x_masked, masks, masked_indices = mask_tokens(x, mask_ratio, self.mask_token) 
        # print("Shape of masked version of img : ", x_masked.shape)
        x = x_masked + self.pos_embed.to(x.device)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x, masked_indices

class Encoder_mae(nn.Module):
    """1 Frame masking mae for weighted masking loss"""
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
        super().__init__(mask_ratio, resolution, channels, pretrained_encoder, patch_size, z_channels, e_dim,
                         normalize_embedding)
        
        self.patch_size = patch_size
        self.image_size = resolution
        self.mask_ratio = mask_ratio        

        self.encoder = VisionTransformerWithPretrainedWts_mae(patch_size=self.patch_size, img_size=self.image_size, 
                                                          mask_ratio=self.mask_ratio)
    
    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        h, masked_indices = self.encoder.forward_features(img, self.mask_ratio)
        h = h.permute(0, 2, 1).contiguous()
        h = h.reshape(h.shape[0], -1, img.size(2)//self.patch_size, img.size(3)//self.patch_size)
        return h, masked_indices

class VisionTransformerWithPretrainedWts_randomMasking(VisionTransformer):
    def __init__(self, patch_size, img_size, mask_ratio, mask_ratio_min, mask_ratio_max, **kwargs):
        """
        pretrained_cfg: pass the same kwargs you’d pass to timm.create_model

        """
        super().__init__(img_size=img_size, **kwargs)
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max

        self.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=patch_size, stride=patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        
    def forward_features(self, x):
        x = self.patch_embed.proj(x)                    
        x = x.flatten(2).transpose(1, 2)   # [B, N, D]
        
        mask_ratio_random = round(random.uniform(self.mask_ratio_min, self.mask_ratio_max), 1)

        x_masked, masks, masked_indices = mask_tokens(x, mask_ratio_random, self.mask_token) 
        x = x_masked + self.pos_embed.to(x.device)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x

class Encoder_randommasking(nn.Module):
    """Encoder : random range masking"""
    def __init__(
        self, 
        mask_ratio,
        mask_ratio_min,
        mask_ratio_max,
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
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        
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
        self.encoder = VisionTransformerWithPretrainedWts_randomMasking(patch_size=self.patch_size, img_size=self.image_size, 
                                                          mask_ratio=self.mask_ratio, mask_ratio_min=self.mask_ratio_min,
                                                          mask_ratio_max=self.mask_ratio_max) 
        state_dict['pos_embed'] = state_dict['pos_embed'][:, 1:, :]

        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)

        print(f"Loaded with {len(missing)} missing keys and {len(unexpected)} unexpected keys")
        print("Missing keys:")
        print(missing)
        print("Unexpected Keys: ")
        print(unexpected)
    
    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        h = self.encoder.forward_features(img)
        h = h.permute(0, 2, 1).contiguous()
        h = h.reshape(h.shape[0], -1, img.size(2)//self.patch_size, img.size(3)//self.patch_size)
        return h

class VisionTransformerWithPretrainedWtsMultiframes(VisionTransformer):
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

    def forward_features(self, x1, x2, mask_ratio):
        x1 = self.patch_embed.proj(x1) 
        x2 = self.patch_embed.proj(x2)   

        x1 = x1.flatten(2).transpose(1, 2)   # [B, N, D]
        x2 = x2.flatten(2).transpose(1, 2)   # [B, N, D]

        x_masked, masks, masked_indices = mask_tokens(x2, mask_ratio, self.mask_token) 
        # print("Shape of masked version of img : ", x_masked.shape)
        
        x1 = x1 + self.pos_embed.to(x1.device)
        x_masked = x_masked + self.pos_embed.to(x_masked.device)

        x = torch.cat([x1, x_masked], dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        x2 = x[:, self.num_patches:]

        return x2
    
class VisionTransformerWithPretrainedWtsMultiframesRandomRange(VisionTransformer):
    def __init__(self, patch_size, img_size, mask_ratio, **kwargs):
        """
        pretrained_cfg: pass the same kwargs you’d pass to timm.create_model

        """
        super().__init__(img_size=img_size,**kwargs)
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        self.mask_ratio_min = 0.0

        self.visible_tokens = int(self.num_patches * (1 - self.mask_ratio))

        self.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=patch_size, stride=patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))

    def forward_features(self, x1, x2, mask_ratio):
        x1 = self.patch_embed.proj(x1) 
        x2 = self.patch_embed.proj(x2)   

        x1 = x1.flatten(2).transpose(1, 2)   # [B, N, D]
        x2 = x2.flatten(2).transpose(1, 2)   # [B, N, D]

        x1 = x1 + self.pos_embed.to(x1.device)

        mask_ratio_random = round(random.uniform(self.mask_ratio_min, self.mask_ratio), 1)

        visible_tokens = int(self.num_patches * (1 - mask_ratio_random))

        x1_masked, masks, masked_indices = mask_tokens_discard(x1, mask_ratio_random)
        x_masked, masks, masked_indices = mask_tokens(x2, mask_ratio_random, self.mask_token) 
               
        x_masked = x_masked + self.pos_embed.to(x_masked.device)

        x = torch.cat([x1_masked, x_masked], dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        x2 = x[:, x1_masked.shape[1]:]

        return x2

class Encoder_multiframes(nn.Module):
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
        self.encoder = VisionTransformerWithPretrainedWtsMultiframesRandomRange(patch_size=self.patch_size, img_size=self.image_size, 
                                                          mask_ratio=self.mask_ratio) 
        state_dict['pos_embed'] = state_dict['pos_embed'][:, 1:, :]

        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)

        print(f"Loaded with {len(missing)} missing keys and {len(unexpected)} unexpected keys")
        print("Missing keys:")
        print(missing)
        print("Unexpected Keys: ")
        print(unexpected)
    
    def forward(self, img1: torch.FloatTensor, img2: torch.FloatTensor) -> torch.FloatTensor:
        h = self.encoder.forward_features(img1, img2, self.mask_ratio)
        h = h.permute(0, 2, 1).contiguous()
        h = h.reshape(h.shape[0], -1, img2.size(2)//self.patch_size, img2.size(3)//self.patch_size)
        return h

class VisionTransformerWithPretrainedWtsMultiframes_2dgrid(VisionTransformer):
    def __init__(self, patch_size, img_size, mask_ratio, **kwargs):
        """
        pretrained_cfg: pass the same kwargs you’d pass to timm.create_model

        """
        super().__init__(img_size=img_size,**kwargs)
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        self.grid_size = patch_size
        self.num_latents = self.grid_size ** 2

        self.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=patch_size, stride=patch_size)
        self.latent_grid = nn.Parameter(torch.zeros(1, self.embed_dim, self.grid_size, self.grid_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, 2*self.num_patches+self.num_latents, self.embed_dim))

    def forward_features(self, x1, x2, mask_ratio):
        B, C, H, W = x1.shape

        x1 = self.patch_embed.proj(x1) 
        x2 = self.patch_embed.proj(x2)   

        x1 = x1.flatten(2).transpose(1, 2)   # [B, N, D]
        x2 = x2.flatten(2).transpose(1, 2)   # [B, N, D]

        grid = self.latent_grid.expand(B, -1, -1, -1)
        grid = grid.flatten(2).transpose(1, 2)

        x = torch.cat([x1, x2, grid], dim=1)
        x = x + self.pos_embed.to(x1.device)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        latent_grid = x[:, self.num_patches+self.num_patches:]

        return latent_grid

class Encoder_multiframes_2dgrid(nn.Module):
    """
        Titok experiment but now with 2d grid like learnable canvas with 2 input frames unmasked
    """
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
        self.num_latent_tokens = patch_size**2
        
        self.init_transformer(pretrained_encoder)

    def init_transformer(self, pretrained_encoder):
        if pretrained_encoder == 'MAE':
            pretrained_encoder_model = 'timm/vit_base_patch16_224.mae'
       
        pretrained_model = timm.create_model(pretrained_encoder_model, img_size=self.image_size, patch_size=self.patch_size, pretrained=True)
        state_dict = pretrained_model.state_dict()
        self.encoder = VisionTransformerWithPretrainedWtsMultiframes_2dgrid(patch_size=self.patch_size, img_size=self.image_size, 
                                                          mask_ratio=self.mask_ratio) 
        
        K = self.num_latent_tokens
        state_dict['pos_embed'] = nn.Parameter(torch.zeros(1, 2*(state_dict['pos_embed'].shape[1]-1)+K, 768))

        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)

        print(f"Loaded with {len(missing)} missing keys and {len(unexpected)} unexpected keys")
        print("Missing keys:")
        print(missing)
        print("Unexpected Keys: ")
        print(unexpected)
    
    def forward(self, img1: torch.FloatTensor, img2: torch.FloatTensor) -> torch.FloatTensor:
        h = self.encoder.forward_features(img1, img2, self.mask_ratio)
        h = h.reshape(h.shape[0], -1, img2.size(2)//self.patch_size, img2.size(3)//self.patch_size)
        return h

class VisionTransformerWithPretrainedWtsMultiframes_tmasked(VisionTransformer):
    def __init__(self, patch_size, img_size, mask_ratio_f1, mask_ratio_f2, **kwargs):
        """
        pretrained_cfg: pass the same kwargs you’d pass to timm.create_model

        """
        super().__init__(img_size=img_size,**kwargs)
        self.num_patches = (img_size // patch_size) ** 2

        self.mask_ratio_f1 = mask_ratio_f1
        self.mask_ratio_f2 = mask_ratio_f2

        self.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=patch_size, stride=patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))

    def forward_features(self, x1, x2):
        x1 = self.patch_embed.proj(x1) 
        x2 = self.patch_embed.proj(x2)   

        x1 = x1.flatten(2).transpose(1, 2)   # [B, N, D]
        x2 = x2.flatten(2).transpose(1, 2)   # [B, N, D]

        x1 = x1 + self.pos_embed.to(x1.device)

        x1_visible, x2_masked = tube_mask_two_frames(x1, x2, self.mask_ratio_f1, self.mask_ratio_f2, self.mask_token) 
        
        x2_masked = x2_masked + self.pos_embed.to(x2_masked.device)

        x = torch.cat([x1_visible, x2_masked], dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        N1_visible = x1_visible.shape[1]
        x2 = x[:, N1_visible:, :]

        return x2

class Encoder_multiframes_tmasked(nn.Module):
    def __init__(
        self, 
        mask_ratio_f1,
        mask_ratio_f2,
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
        self.mask_ratio_f1 = mask_ratio_f1
        self.mask_ratio_f2 = mask_ratio_f2
        
        if pretrained_encoder == 'MAE':
            pretrained_encoder_model = 'timm/vit_base_patch16_224.mae'
       
        # self.encoder = timm.create_model(pretrained_encoder_model, img_size=self.image_size, patch_size=self.patch_size, pretrained=False, dynamic_img_size=True).train()
        pretrained_model = timm.create_model(pretrained_encoder_model, img_size=self.image_size, patch_size=self.patch_size, pretrained=True)
        state_dict = pretrained_model.state_dict()

        self.encoder = VisionTransformerWithPretrainedWtsMultiframes_tmasked(patch_size=self.patch_size, img_size=self.image_size, 
                                                          mask_ratio_f1=self.mask_ratio_f1, mask_ratio_f2=self.mask_ratio_f2) 
        
        state_dict['pos_embed'] = nn.Parameter(torch.zeros(1, (state_dict['pos_embed'].shape[1]-1), 768))

        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)

        print(f"Loaded with {len(missing)} missing keys and {len(unexpected)} unexpected keys")
        print("Missing keys:")
        print(missing)
        print("Unexpected Keys: ")
        print(unexpected)
    
    def forward(self, img1: torch.FloatTensor, img2: torch.FloatTensor) -> torch.FloatTensor:
       
        h = self.encoder.forward_features(img1, img2) # [B, N, D]
        h = h.permute(0, 2, 1).contiguous()
        h = h.reshape(h.shape[0], -1, img2.size(2)//self.patch_size, img2.size(3)//self.patch_size)
        return h