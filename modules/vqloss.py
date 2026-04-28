import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.discriminator import NLayerDiscriminator, weights_init
from .lpips import LPIPS 

def adopt_weight(weight, global_step, threshold=0, value=0.):
    return weight if global_step >= threshold else value

def hinge_d_loss(logits_real, logits_fake):
    return 0.5 * (F.relu(1. - logits_real).mean() + F.relu(1. + logits_fake).mean())

def vanilla_d_loss(logits_real, logits_fake):
    return 0.5 * (
        F.softplus(-logits_real).mean() + F.softplus(logits_fake).mean()
    )

class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()

class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        entropy_loss_weight=1.0,
        distill_loss_weight=0.1,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        adaptive_disc_weight=True,
        l1_loss_weight=1.0,
        l2_loss_weight=0.0,
        perceptual_weight=1.0,
        se_weight=0.25,
        use_actnorm=False,
        disc_conditional=False,
        disc_ndf=64,
        disc_loss="hinge",
        warmup_steps=1000,
        beta_1=0.5,
        beta_2=0.9
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"], f"Invalid disc_loss: {disc_loss}"

        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.distill_loss_weight = distill_loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.l2_loss_weight = l2_loss_weight
        self.perceptual_weight = perceptual_weight
        self.adaptive_disc_weight = adaptive_disc_weight
        self.disc_factor = disc_factor
        self.disc_weight = disc_weight
        self.se_weight = se_weight

        self.warmup_steps = warmup_steps
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.disc_conditional = disc_conditional
        self.discriminator_iter_start = disc_start

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm,
            ndf=disc_ndf
        ).apply(weights_init)
        
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        print(f"{self.__class__.__name__} initialized with {disc_loss} loss.")

        # Perceptual loss
        self.perceptual_loss = LPIPS().eval()

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            return torch.tensor(0.1)

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.disc_weight
        return d_weight

    def forward(
        self, codebook_entropy_losses, distill_loss, inputs, reconstructions,
        optimizer_idx, global_step, last_layer=None, cond=None, split="train"
    ):
        # codebook_loss, entropy_loss = codebook_entropy_losses
        codebook_entropy_losses = None
        gan_loss = None

        if optimizer_idx == 0:
            # Reconstruction and Perceptual Loss
            # rec_loss = torch.tensor(0.0, device=distill_loss.device)
            # p_loss = torch.tensor(0.0, device=distill_loss.device)
            
            if isinstance(inputs, list):
                for idx, (input, recon) in enumerate(zip(inputs, reconstructions)):
                    se_weight = 1 if idx == 0 else self.se_weight
                    l1 = torch.abs(input - recon).mean()
                    l2 = F.mse_loss(recon, input)
                    rec_loss += (self.l1_loss_weight * l1 + self.l2_loss_weight * l2) * se_weight
                    if self.perceptual_weight > 0:
                        p_loss += self.perceptual_loss(input, recon).mean() * se_weight
            else:
                l1 = torch.abs(inputs - reconstructions).mean()
                l2 = F.mse_loss(reconstructions, inputs)
                rec_loss = self.l1_loss_weight * l1 + self.l2_loss_weight * l2
                if self.perceptual_weight > 0:
                    p_loss = self.perceptual_loss(inputs, reconstructions).mean()
                    
            nll_loss = rec_loss + self.perceptual_weight * p_loss

            reconstruction = reconstructions[0] if isinstance(reconstructions, list) else reconstructions
            
            if gan_loss is not None:
                if self.disc_conditional and cond is not None:
                    logits_fake = self.discriminator(torch.cat((reconstruction, cond), dim=1))
                else:
                    logits_fake = self.discriminator(reconstruction)

                g_loss = -torch.mean(logits_fake)

                if self.adaptive_disc_weight:
                    try:
                        d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer)
                    except RuntimeError:
                        assert not self.training
                        d_weight = torch.tensor(0.0, device=distill_loss.device)
                else:
                    d_weight = self.disc_weight

                disc_factor = adopt_weight(self.disc_factor, global_step, self.discriminator_iter_start)
                loss = nll_loss + d_weight * disc_factor * g_loss
                
            loss = nll_loss
            # if codebook_entropy_losses is not None:
            #     loss += self.codebook_weight * codebook_loss.mean()
            #     if entropy_loss is not None:
            #         loss += self.entropy_loss_weight * entropy_loss.mean()
                # loss += self.distill_loss_weight * distill_loss.mean()

            log = {
                f"{split}/total_loss": loss.detach().mean(),
                # f"{split}/quant_loss": codebook_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/p_loss": p_loss.detach().mean(),
            }
            
            return loss, log

        if gan_loss is not None:
            if optimizer_idx == 1:
                if isinstance(inputs, list):
                    inputs = inputs[0]
                    reconstructions = reconstructions[0]
                # Discriminator 
                if self.disc_conditional and cond is not None:
                    real = torch.cat((inputs.detach(), cond), dim=1)
                    fake = torch.cat((reconstructions.detach(), cond), dim=1)
                else:
                    real = inputs.detach()
                    fake = reconstructions.detach()

                logits_real = self.discriminator(real)
                logits_fake = self.discriminator(fake)

                disc_factor = adopt_weight(self.disc_factor, global_step, self.discriminator_iter_start)
                d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

                log = {
                    f"{split}/disc_loss": d_loss.detach().mean(),
                    f"{split}/logits_real": logits_real.detach().mean(),
                    f"{split}/logits_fake": logits_fake.detach().mean(),
                }
                return d_loss, log
            
class VQLPIPSWithDiscriminator_decoderbased(VQLPIPSWithDiscriminator):
    def __init__(
        self,
        disc_start,
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        entropy_loss_weight=1.0,
        distill_loss_weight=0.1,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        adaptive_disc_weight=True,
        l1_loss_weight=1.0,
        l2_loss_weight=0.0,
        perceptual_weight=1.0,
        jepa_loss_weight=1.0,
        se_weight=0.25,
        use_actnorm=False,
        disc_conditional=False,
        disc_ndf=64,
        disc_loss="hinge",
        warmup_steps=1000,
        beta_1=0.5,
        beta_2=0.9
    ):
        super().__init__(disc_start, codebook_weight, pixelloss_weight, entropy_loss_weight, distill_loss_weight, disc_num_layers, 
                         disc_in_channels, disc_factor, disc_weight, adaptive_disc_weight, l1_loss_weight, l2_loss_weight, 
                         perceptual_weight, se_weight, use_actnorm, disc_conditional, disc_ndf, disc_loss, warmup_steps, 
                         beta_1, beta_2)
        assert disc_loss in ["hinge", "vanilla"], f"Invalid disc_loss: {disc_loss}"

        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.jepa_loss_weight = jepa_loss_weight

        # Perceptual loss
        self.perceptual_loss = LPIPS().eval()

    def forward(
        self, codebook_entropy_losses, distill_loss, inputs, reconstructions, target, dec_target,
        optimizer_idx, global_step, last_layer=None, cond=None, split="train"
    ):
        codebook_loss, entropy_loss = codebook_entropy_losses
        gan_loss = None

        if optimizer_idx == 0:
            # Reconstruction and Perceptual Loss
            # rec_loss = torch.tensor(0.0, device=distill_loss.device)
            # p_loss = torch.tensor(0.0, device=distill_loss.device)
            
            if isinstance(inputs, list):
                for idx, (input, recon) in enumerate(zip(inputs, reconstructions)):
                    se_weight = 1 if idx == 0 else self.se_weight
                    l1 = torch.abs(input - recon).mean()
                    l2 = F.mse_loss(recon, input)
                    rec_loss += (self.l1_loss_weight * l1 + self.l2_loss_weight * l2) * se_weight
                    if self.perceptual_weight > 0:
                        p_loss += self.perceptual_loss(input, recon).mean() * se_weight
            else:
                l1 = torch.abs(inputs - reconstructions).mean()
                l2 = F.mse_loss(reconstructions, inputs)
                rec_loss = self.l1_loss_weight * l1 + self.l2_loss_weight * l2
                if self.perceptual_weight > 0:
                    p_loss = self.perceptual_loss(inputs, reconstructions).mean()

            cos_sim = F.cosine_similarity(dec_target, target, dim=-1)
            jepa_loss = (1 - cos_sim).mean()
                    
            nll_loss = rec_loss + self.perceptual_weight * p_loss
            nll_loss += self.jepa_loss_weight * jepa_loss

            reconstruction = reconstructions[0] if isinstance(reconstructions, list) else reconstructions
                
            loss = nll_loss
            loss += self.codebook_weight * codebook_loss.mean()
            if entropy_loss is not None:
                loss += self.entropy_loss_weight * entropy_loss.mean()
            # loss += self.distill_loss_weight * distill_loss.mean()

            log = {
                f"{split}/total_loss": loss.detach().mean(),
                f"{split}/quant_loss": codebook_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/p_loss": p_loss.detach().mean(),
                f"{split}/jepa_loss": jepa_loss.detach().mean(),
            }
            if entropy_loss is not None:
                log[f"{split}/entropy_loss"] = entropy_loss.detach().mean()
                log[f"{split}/entropy_loss_weight"] = torch.tensor(self.entropy_loss_weight)

            return loss, log


class VQLPIPSWithDiscriminator_mae(VQLPIPSWithDiscriminator):
    """Loss weighing for masking"""
    def __init__(
        self,
        disc_start,
        mask_loss_wt,
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        entropy_loss_weight=1.0,
        distill_loss_weight=0.1,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        adaptive_disc_weight=True,
        l1_loss_weight=1.0,
        l2_loss_weight=0.0,
        perceptual_weight=1.0,
        se_weight=0.25,
        use_actnorm=False,
        disc_conditional=False,
        disc_ndf=64,
        disc_loss="hinge",
        warmup_steps=1000,
        beta_1=0.5,
        beta_2=0.9
    ):
        super().__init__(disc_start, codebook_weight, pixelloss_weight, entropy_loss_weight, distill_loss_weight, disc_num_layers, 
                         disc_in_channels, disc_factor, disc_weight, adaptive_disc_weight, l1_loss_weight, l2_loss_weight, 
                         perceptual_weight, se_weight, use_actnorm, disc_conditional, disc_ndf, disc_loss, warmup_steps, 
                         beta_1, beta_2)
        assert disc_loss in ["hinge", "vanilla"], f"Invalid disc_loss: {disc_loss}"

        self.mask_loss_wt = mask_loss_wt
        # Perceptual loss
        self.perceptual_loss = LPIPS().eval()

    def masked_tokens(self, input, mask_indices, patch_size=16):
        B, C, H, W = input.shape

        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        N = num_patches_h * num_patches_w

        # make patches
        patches = input.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, N, C, patch_size, patch_size)

        # select masked patches
        masked_patches = patches[torch.arange(B)[:, None], mask_indices]

        return masked_patches

    def forward(
        self, codebook_entropy_losses, distill_loss, inputs, reconstructions, masked_indices,
        optimizer_idx, global_step, last_layer=None, cond=None, split="train"
    ):
        codebook_loss, entropy_loss = codebook_entropy_losses
        gan_loss = None

        if optimizer_idx == 0:
            # Reconstruction and Perceptual Loss
            # rec_loss = torch.tensor(0.0, device=distill_loss.device)
            # p_loss = torch.tensor(0.0, device=distill_loss.device)
            input_masks = self.masked_tokens(inputs, masked_indices, patch_size=16)
            reconstruction_masks = self.masked_tokens(reconstructions, masked_indices, patch_size=16)
            
            if isinstance(inputs, list):
                for idx, (input, recon) in enumerate(zip(inputs, reconstructions)):
                    se_weight = 1 if idx == 0 else self.se_weight
                    l1 = torch.abs(input - recon).mean()
                    l2 = F.mse_loss(recon, input)
                    rec_loss += (self.l1_loss_weight * l1 + self.l2_loss_weight * l2) * se_weight
                    if self.perceptual_weight > 0:
                        p_loss += self.perceptual_loss(input, recon).mean() * se_weight
            else:
                l1 = torch.abs(inputs - reconstructions).mean()
                l2 = F.mse_loss(reconstructions, inputs)
                rec_loss = self.l1_loss_weight * l1 + self.l2_loss_weight * l2
                if self.perceptual_weight > 0:
                    p_loss = self.perceptual_loss(inputs, reconstructions).mean()

                l1_masked = torch.abs(input_masks - reconstruction_masks).mean()
                p_loss_masked = self.perceptual_loss(input_masks, reconstruction_masks).mean()
                mask_recon_loss = (self.l1_loss_weight * l1_masked) + (self.perceptual_weight * p_loss_masked)
          
            nll_loss = rec_loss + self.perceptual_weight * p_loss
            nll_loss += self.mask_loss_wt * (mask_recon_loss)

            reconstruction = reconstructions[0] if isinstance(reconstructions, list) else reconstructions
                
            loss = nll_loss
            loss += self.codebook_weight * codebook_loss.mean()
            if entropy_loss is not None:
                loss += self.entropy_loss_weight * entropy_loss.mean()
            # loss += self.distill_loss_weight * distill_loss.mean()

            log = {
                f"{split}/total_loss": loss.detach().mean(),
                f"{split}/quant_loss": codebook_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/mask_rec_loss": mask_recon_loss.detach().mean(),
                f"{split}/p_loss": p_loss.detach().mean(),
            }
            if entropy_loss is not None:
                log[f"{split}/entropy_loss"] = entropy_loss.detach().mean()
                log[f"{split}/entropy_loss_weight"] = torch.tensor(self.entropy_loss_weight)

            return loss, log

class VQLPIPSWithDiscriminator_motion(VQLPIPSWithDiscriminator):
    def __init__(
        self,
        disc_start,
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        entropy_loss_weight=1.0,
        entropy_loss_weight_q2=1.0,
        distill_loss_weight=0.1,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        adaptive_disc_weight=True,
        l1_loss_weight=1.0,
        l2_loss_weight=0.0,
        perceptual_weight=1.0,
        se_weight=0.25,
        use_actnorm=False,
        disc_conditional=False,
        disc_ndf=64,
        disc_loss="hinge",
        warmup_steps=1000,
        beta_1=0.5,
        beta_2=0.9
    ):
        super().__init__(disc_start, codebook_weight, pixelloss_weight, entropy_loss_weight, distill_loss_weight, disc_num_layers, 
                         disc_in_channels, disc_factor, disc_weight, adaptive_disc_weight, l1_loss_weight, l2_loss_weight, 
                         perceptual_weight, se_weight, use_actnorm, disc_conditional, disc_ndf, disc_loss, warmup_steps, 
                         beta_1, beta_2)

        self.l1_loss_weight = l1_loss_weight
        self.l2_loss_weight = l2_loss_weight
        self.perceptual_weight = perceptual_weight
        self.codebook_weight = codebook_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_weight_q2 = entropy_loss_weight_q2
        
        self.perceptual_loss = LPIPS().eval()
    
    def forward(
        self, codebook_entropy_losses1, codebook_entropy_losses2, distill_loss, inputs, reconstructions, gt_displacement, pred_displacement,
        optimizer_idx, global_step, last_layer=None, cond=None, split="train"
    ):
        codebook_loss1, entropy_loss1 = codebook_entropy_losses1
        codebook_loss2, entropy_loss2 = codebook_entropy_losses2
        gan_loss = None

        if optimizer_idx == 0:
            # Reconstruction and Perceptual Loss
            # rec_loss = torch.tensor(0.0, device=distill_loss.device)
            # p_loss = torch.tensor(0.0, device=distill_loss.device)
            
            if isinstance(inputs, list):
                for idx, (input, recon) in enumerate(zip(inputs, reconstructions)):
                    se_weight = 1 if idx == 0 else self.se_weight
                    l1 = torch.abs(input - recon).mean()
                    l2 = F.mse_loss(recon, input)
                    rec_loss += (self.l1_loss_weight * l1 + self.l2_loss_weight * l2) * se_weight
                    if self.perceptual_weight > 0:
                        p_loss += self.perceptual_loss(input, recon).mean() * se_weight
            else:
                l1 = torch.abs(inputs - reconstructions).mean()
                l2 = F.mse_loss(reconstructions, inputs)
                rec_loss = self.l1_loss_weight * l1 + self.l2_loss_weight * l2
                if self.perceptual_weight > 0:
                    p_loss = self.perceptual_loss(inputs, reconstructions).mean()
                    
            nll_loss = rec_loss + self.perceptual_weight * p_loss

            l2_motion = F.mse_loss(pred_displacement, gt_displacement)

            reconstruction = reconstructions[0] if isinstance(reconstructions, list) else reconstructions
                
            loss = nll_loss + l2_motion
            loss += (self.codebook_weight * codebook_loss1.mean()) + (self.codebook_weight * codebook_loss2.mean())
            if entropy_loss1 is not None:
                loss += (self.entropy_loss_weight * entropy_loss1.mean()) + (self.entropy_loss_weight_q2 * entropy_loss2.mean())
            # loss += self.distill_loss_weight * distill_loss.mean()

            log = {
                f"{split}/total_loss": loss.detach().mean(),
                f"{split}/quant_loss1": codebook_loss1.detach().mean(),
                f"{split}/quant_loss2": codebook_loss2.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/p_loss": p_loss.detach().mean(),
                f"{split}/disp_loss": l2_motion.detach().mean(),
            }
            if entropy_loss1 is not None:
                log[f"{split}/entropy_loss1"] = entropy_loss1.detach().mean()
                log[f"{split}/entropy_loss2"] = entropy_loss2.detach().mean()
                log[f"{split}/entropy_loss_weight"] = torch.tensor(self.entropy_loss_weight)
                log[f"{split}/entropy_loss_weight_q2"] = torch.tensor(self.entropy_loss_weight_q2)

            return loss, log

class VQLPIPSWithDiscriminator_depth(VQLPIPSWithDiscriminator):
    def __init__(
        self,
        disc_start,
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        entropy_loss_weight=1.0,
        entropy_loss_weight_q2=1.0,
        distill_loss_weight=0.1,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        adaptive_disc_weight=True,
        l1_loss_weight=1.0,
        l2_loss_weight=0.0,
        perceptual_weight=1.0,
        se_weight=0.25,
        use_actnorm=False,
        disc_conditional=False,
        disc_ndf=64,
        disc_loss="hinge",
        warmup_steps=1000,
        beta_1=0.5,
        beta_2=0.9
    ):
        super().__init__(disc_start, codebook_weight, pixelloss_weight, entropy_loss_weight, distill_loss_weight, disc_num_layers, 
                         disc_in_channels, disc_factor, disc_weight, adaptive_disc_weight, l1_loss_weight, l2_loss_weight, 
                         perceptual_weight, se_weight, use_actnorm, disc_conditional, disc_ndf, disc_loss, warmup_steps, 
                         beta_1, beta_2)

        self.l1_loss_weight = l1_loss_weight
        self.l2_loss_weight = l2_loss_weight
        self.perceptual_weight = perceptual_weight
        self.codebook_weight = codebook_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_weight_q2 = entropy_loss_weight_q2
        
        self.perceptual_loss = LPIPS().eval()
    
    def forward(
        self, codebook_entropy_losses1, codebook_entropy_losses2, distill_loss, inputs, reconstructions, gt_depth, pred_depth,
        optimizer_idx, global_step, last_layer=None, cond=None, split="train"
    ):
        codebook_loss1, entropy_loss1 = codebook_entropy_losses1
        codebook_loss2, entropy_loss2 = codebook_entropy_losses2
        gan_loss = None

        if optimizer_idx == 0:
            if isinstance(inputs, list):
                for idx, (input, recon) in enumerate(zip(inputs, reconstructions)):
                    se_weight = 1 if idx == 0 else self.se_weight
                    l1 = torch.abs(input - recon).mean()
                    l2 = F.mse_loss(recon, input)
                    rec_loss += (self.l1_loss_weight * l1 + self.l2_loss_weight * l2) * se_weight
                    if self.perceptual_weight > 0:
                        p_loss += self.perceptual_loss(input, recon).mean() * se_weight
            else:
                l1 = torch.abs(inputs - reconstructions).mean()
                l2 = F.mse_loss(reconstructions, inputs)
                rec_loss = self.l1_loss_weight * l1 + self.l2_loss_weight * l2
                if self.perceptual_weight > 0:
                    p_loss = self.perceptual_loss(inputs, reconstructions).mean()
                    
            nll_loss = rec_loss + self.perceptual_weight * p_loss

            l1_depth = F.l1_loss(pred_depth, gt_depth)

            reconstruction = reconstructions[0] if isinstance(reconstructions, list) else reconstructions
                
            loss = nll_loss + l1_depth
            loss += (self.codebook_weight * codebook_loss1.mean()) + (self.codebook_weight * codebook_loss2.mean())
            if entropy_loss1 is not None:
                loss += (self.entropy_loss_weight * entropy_loss1.mean()) + (self.entropy_loss_weight_q2 * entropy_loss2.mean())
            # loss += self.distill_loss_weight * distill_loss.mean()

            log = {
                f"{split}/total_loss": loss.detach().mean(),
                f"{split}/quant_loss1": codebook_loss1.detach().mean(),
                f"{split}/quant_loss2": codebook_loss2.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/p_loss": p_loss.detach().mean(),
                f"{split}/depth_loss": l1_depth.detach().mean(),
                # f"{split}/d_weight": d_weight,
                # f"{split}/disc_factor": torch.tensor(disc_factor),
                # f"{split}/g_loss": g_loss.detach().mean(),
            }
            if entropy_loss1 is not None:
                log[f"{split}/entropy_loss1"] = entropy_loss1.detach().mean()
                log[f"{split}/entropy_loss2"] = entropy_loss2.detach().mean()
                log[f"{split}/entropy_loss_weight"] = torch.tensor(self.entropy_loss_weight)
                log[f"{split}/entropy_loss_weight_q2"] = torch.tensor(self.entropy_loss_weight_q2)

            return loss, log

class VQLPIPSWithDiscriminator_dino(VQLPIPSWithDiscriminator):
    def __init__(
        self,
        disc_start,
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        entropy_loss_weight=1.0,
        entropy_loss_weight_q2=1.0,
        distill_loss_weight=0.1,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        dino_wt=1.0,
        adaptive_disc_weight=True,
        l1_loss_weight=1.0,
        l2_loss_weight=0.0,
        perceptual_weight=1.0,
        se_weight=0.25,
        use_actnorm=False,
        disc_conditional=False,
        disc_ndf=64,
        disc_loss="hinge",
        warmup_steps=1000,
        beta_1=0.5,
        beta_2=0.9
    ):
        super().__init__(disc_start, codebook_weight, pixelloss_weight, entropy_loss_weight, distill_loss_weight, disc_num_layers, 
                         disc_in_channels, disc_factor, disc_weight, adaptive_disc_weight, l1_loss_weight, l2_loss_weight, 
                         perceptual_weight, se_weight, use_actnorm, disc_conditional, disc_ndf, disc_loss, warmup_steps, 
                         beta_1, beta_2)

        self.l1_loss_weight = l1_loss_weight
        self.l2_loss_weight = l2_loss_weight
        self.perceptual_weight = perceptual_weight
        self.codebook_weight = codebook_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_weight_q2 = entropy_loss_weight_q2
        self.dino_wt = dino_wt
        
        self.perceptual_loss = LPIPS().eval()
    
    def forward(
        self, codebook_entropy_losses1, codebook_entropy_losses2, distill_loss, inputs, reconstructions, gt_dino, pred_dino,
        optimizer_idx, global_step, last_layer=None, cond=None, split="train"
    ):
        codebook_loss1, entropy_loss1 = codebook_entropy_losses1
        codebook_loss2, entropy_loss2 = codebook_entropy_losses2
        gan_loss = None

        if optimizer_idx == 0:
            if isinstance(inputs, list):
                for idx, (input, recon) in enumerate(zip(inputs, reconstructions)):
                    se_weight = 1 if idx == 0 else self.se_weight
                    l1 = torch.abs(input - recon).mean()
                    l2 = F.mse_loss(recon, input)
                    rec_loss += (self.l1_loss_weight * l1 + self.l2_loss_weight * l2) * se_weight
                    if self.perceptual_weight > 0:
                        p_loss += self.perceptual_loss(input, recon).mean() * se_weight
            else:
                l1 = torch.abs(inputs - reconstructions).mean()
                l2 = F.mse_loss(reconstructions, inputs)
                rec_loss = self.l1_loss_weight * l1 + self.l2_loss_weight * l2
                if self.perceptual_weight > 0:
                    p_loss = self.perceptual_loss(inputs, reconstructions).mean()
                    
            nll_loss = rec_loss + self.perceptual_weight * p_loss

            loss_dist = (1 - (pred_dino * gt_dino).sum(-1)).mean()

            reconstruction = reconstructions[0] if isinstance(reconstructions, list) else reconstructions
                
            loss = nll_loss + (self.dino_wt * loss_dist)
            loss += (self.codebook_weight * codebook_loss1.mean()) + (self.codebook_weight * codebook_loss2.mean())
            if entropy_loss1 is not None:
                loss += (self.entropy_loss_weight * entropy_loss1.mean()) + (self.entropy_loss_weight_q2 * entropy_loss2.mean())
            # loss += self.distill_loss_weight * distill_loss.mean()

            log = {
                f"{split}/total_loss": loss.detach().mean(),
                f"{split}/quant_loss1": codebook_loss1.detach().mean(),
                f"{split}/quant_loss2": codebook_loss2.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/p_loss": p_loss.detach().mean(),
                f"{split}/distillation_loss": loss_dist.detach().mean(),
                # f"{split}/d_weight": d_weight,
                # f"{split}/disc_factor": torch.tensor(disc_factor),
                # f"{split}/g_loss": g_loss.detach().mean(),
            }
            if entropy_loss1 is not None:
                log[f"{split}/entropy_loss1"] = entropy_loss1.detach().mean()
                log[f"{split}/entropy_loss2"] = entropy_loss2.detach().mean()
                log[f"{split}/entropy_loss_weight"] = torch.tensor(self.entropy_loss_weight)
                log[f"{split}/entropy_loss_weight_q2"] = torch.tensor(self.entropy_loss_weight_q2)

            return loss, log

class VQLPIPSWithDiscriminator_2dgrid(VQLPIPSWithDiscriminator):
    def __init__(
        self,
        disc_start,
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        entropy_loss_weight=1.0,
        distill_loss_weight=0.1,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        adaptive_disc_weight=True,
        l1_loss_weight=1.0,
        l2_loss_weight=0.0,
        perceptual_weight=1.0,
        se_weight=0.25,
        use_actnorm=False,
        disc_conditional=False,
        disc_ndf=64,
        disc_loss="hinge",
        warmup_steps=1000,
        beta_1=0.5,
        beta_2=0.9
    ):
        super().__init__(disc_start, codebook_weight, pixelloss_weight, entropy_loss_weight, distill_loss_weight, disc_num_layers, 
                         disc_in_channels, disc_factor, disc_weight, adaptive_disc_weight, l1_loss_weight, l2_loss_weight, 
                         perceptual_weight, se_weight, use_actnorm, disc_conditional, disc_ndf, disc_loss, warmup_steps, 
                         beta_1, beta_2)

        self.l1_loss_weight = l1_loss_weight
        self.l2_loss_weight = l2_loss_weight
        self.perceptual_weight = perceptual_weight
        self.codebook_weight = codebook_weight
        self.entropy_loss_weight = entropy_loss_weight
        
        self.perceptual_loss = LPIPS().eval()
    
    def forward(
        self, codebook_entropy_losses, distill_loss, input1, input2, reconstruction1, reconstruction2, 
        optimizer_idx, global_step, last_layer=None, cond=None, split="train"
    ):
        codebook_loss, entropy_loss = codebook_entropy_losses
        gan_loss = None

        if optimizer_idx == 0:
            
            l1_f1 = torch.abs(input1 - reconstruction1).mean()
            l2_f1 = F.mse_loss(reconstruction1, input1)

            l1_f2 = torch.abs(input2 - reconstruction2).mean()
            l2_f2 = F.mse_loss(reconstruction2, input2)

            rec_loss_f1 = self.l1_loss_weight * l1_f1 + self.l2_loss_weight * l2_f1
            rec_loss_f2 = self.l1_loss_weight * l1_f2 + self.l2_loss_weight * l2_f2

            rec_loss = (rec_loss_f1 + rec_loss_f2) / 2

            if self.perceptual_weight > 0:
                p_loss_f1 = self.perceptual_loss(input1, reconstruction1).mean()
                p_loss_f2 = self.perceptual_loss(input2, reconstruction2).mean()
                p_loss = (p_loss_f1 + p_loss_f2) / 2
                    
            nll_loss = rec_loss + self.perceptual_weight * p_loss
                
            loss = nll_loss
            loss += self.codebook_weight * codebook_loss.mean()
            if entropy_loss is not None:
                loss += self.entropy_loss_weight * entropy_loss.mean()
            # loss += self.distill_loss_weight * distill_loss.mean()

            log = {
                f"{split}/total_loss": loss.detach().mean(),
                f"{split}/quant_loss": codebook_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss_f1": rec_loss_f1.detach().mean(),
                f"{split}/rec_loss_f2": rec_loss_f2.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/p_loss_f1": p_loss_f1.detach().mean(),
                f"{split}/p_loss_f2": p_loss_f2.detach().mean(),
                f"{split}/p_loss": p_loss.detach().mean(),
            }
            if entropy_loss is not None:
                log[f"{split}/entropy_loss"] = entropy_loss.detach().mean()
                log[f"{split}/entropy_loss_weight"] = torch.tensor(self.entropy_loss_weight)

            return loss, log

class VQLPIPSWithDiscriminator_jepa(VQLPIPSWithDiscriminator):
    def __init__(
        self,
        disc_start,
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        entropy_loss_weight=1.0,
        entropy_loss_weight_q2=1.0,
        distill_loss_weight=0.1,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        adaptive_disc_weight=True,
        l1_loss_weight=1.0,
        l2_loss_weight=0.0,
        perceptual_weight=1.0,
        se_weight=0.25,
        use_actnorm=False,
        disc_conditional=False,
        disc_ndf=64,
        disc_loss="hinge",
        warmup_steps=1000,
        beta_1=0.5,
        beta_2=0.9,
        cosine_loss_weight=1.0,
    ):
        super().__init__(disc_start, codebook_weight, pixelloss_weight, entropy_loss_weight, distill_loss_weight, disc_num_layers, 
                         disc_in_channels, disc_factor, disc_weight, adaptive_disc_weight, l1_loss_weight, l2_loss_weight, 
                         perceptual_weight, se_weight, use_actnorm, disc_conditional, disc_ndf, disc_loss, warmup_steps, 
                         beta_1, beta_2)

        self.l1_loss_weight = l1_loss_weight
        self.l2_loss_weight = l2_loss_weight
        self.perceptual_weight = perceptual_weight
        self.codebook_weight = codebook_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_weight_q2 = entropy_loss_weight_q2
        self.cosine_loss_weight = cosine_loss_weight
        
        self.perceptual_loss = LPIPS().eval()
    
    def forward(
        self, codebook_entropy_losses1, codebook_entropy_losses2, distill_loss, inputs, reconstructions, target, dec_target,
        optimizer_idx, global_step, last_layer=None, cond=None, split="train"
    ):
        codebook_loss1, entropy_loss1 = codebook_entropy_losses1
        codebook_loss2, entropy_loss2 = codebook_entropy_losses2
        gan_loss = None

        if optimizer_idx == 0:
            if isinstance(inputs, list):
                for idx, (input, recon) in enumerate(zip(inputs, reconstructions)):
                    se_weight = 1 if idx == 0 else self.se_weight
                    l1 = torch.abs(input - recon).mean()
                    l2 = F.mse_loss(recon, input)
                    rec_loss += (self.l1_loss_weight * l1 + self.l2_loss_weight * l2) * se_weight
                    if self.perceptual_weight > 0:
                        p_loss += self.perceptual_loss(input, recon).mean() * se_weight
            else:
                l1 = torch.abs(inputs - reconstructions).mean()
                l2 = F.mse_loss(reconstructions, inputs)
                rec_loss = self.l1_loss_weight * l1 + self.l2_loss_weight * l2
                if self.perceptual_weight > 0:
                    p_loss = self.perceptual_loss(inputs, reconstructions).mean()
                    
            nll_loss = rec_loss + self.perceptual_weight * p_loss

            cos_sim = F.cosine_similarity(dec_target, target, dim=-1)
            loss_jepa = (1 - cos_sim).mean()

            reconstruction = reconstructions[0] if isinstance(reconstructions, list) else reconstructions
                
            loss = nll_loss + (self.cosine_loss_weight * loss_jepa)
            loss += (self.codebook_weight * codebook_loss1.mean()) + (self.codebook_weight * codebook_loss2.mean())
            if entropy_loss1 is not None:
                loss += (self.entropy_loss_weight * entropy_loss1.mean()) + (self.entropy_loss_weight_q2 * entropy_loss2.mean())
            # loss += self.distill_loss_weight * distill_loss.mean()

            log = {
                f"{split}/total_loss": loss.detach().mean(),
                f"{split}/quant_loss1": codebook_loss1.detach().mean(),
                f"{split}/quant_loss2": codebook_loss2.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/p_loss": p_loss.detach().mean(),
                f"{split}/jepa_loss": loss_jepa.detach().mean(),
            }
            if entropy_loss1 is not None:
                log[f"{split}/entropy_loss1"] = entropy_loss1.detach().mean()
                log[f"{split}/entropy_loss2"] = entropy_loss2.detach().mean()
                log[f"{split}/entropy_loss_weight"] = torch.tensor(self.entropy_loss_weight)
                log[f"{split}/entropy_loss_weight_q2"] = torch.tensor(self.entropy_loss_weight_q2)

            return loss, log

class VQLPIPSWithDiscriminator_2dgrid_dino(VQLPIPSWithDiscriminator):
    def __init__(
        self,
        disc_start,
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        entropy_loss_weight=1.0,
        entropy_loss_weight_q2=1.0,
        distill_loss_weight=0.1,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        adaptive_disc_weight=True,
        l1_loss_weight=1.0,
        l2_loss_weight=0.0,
        perceptual_weight=1.0,
        se_weight=0.25,
        use_actnorm=False,
        disc_conditional=False,
        disc_ndf=64,
        disc_loss="hinge",
        warmup_steps=1000,
        beta_1=0.5,
        beta_2=0.9
    ):
        super().__init__(disc_start, codebook_weight, pixelloss_weight, entropy_loss_weight, distill_loss_weight, disc_num_layers, 
                         disc_in_channels, disc_factor, disc_weight, adaptive_disc_weight, l1_loss_weight, l2_loss_weight, 
                         perceptual_weight, se_weight, use_actnorm, disc_conditional, disc_ndf, disc_loss, warmup_steps, 
                         beta_1, beta_2)

        self.l1_loss_weight = l1_loss_weight
        self.l2_loss_weight = l2_loss_weight
        self.perceptual_weight = perceptual_weight
        self.codebook_weight = codebook_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_weight_q2 = entropy_loss_weight_q2
        
        self.perceptual_loss = LPIPS().eval()
    
    def forward(
        self, codebook_entropy_losses1, codebook_entropy_losses2, distill_loss, input1, input2, reconstruction1, reconstruction2,
        gt_dino, pred_dino, optimizer_idx, global_step, last_layer=None, cond=None, split="train"):

        codebook_loss1, entropy_loss1 = codebook_entropy_losses1
        codebook_loss2, entropy_loss2 = codebook_entropy_losses2
        gan_loss = None

        if optimizer_idx == 0:
            
            l1_f1 = torch.abs(input1 - reconstruction1).mean()
            l2_f1 = F.mse_loss(reconstruction1, input1)

            l1_f2 = torch.abs(input2 - reconstruction2).mean()
            l2_f2 = F.mse_loss(reconstruction2, input2)

            rec_loss_f1 = self.l1_loss_weight * l1_f1 + self.l2_loss_weight * l2_f1
            rec_loss_f2 = self.l1_loss_weight * l1_f2 + self.l2_loss_weight * l2_f2

            rec_loss = (rec_loss_f1 + rec_loss_f2) / 2

            if self.perceptual_weight > 0:
                p_loss_f1 = self.perceptual_loss(input1, reconstruction1).mean()
                p_loss_f2 = self.perceptual_loss(input2, reconstruction2).mean()
                p_loss = (p_loss_f1 + p_loss_f2) / 2
                    
            nll_loss = rec_loss + self.perceptual_weight * p_loss
            loss_dino = (1 - (pred_dino * gt_dino).sum(-1)).mean()
                
            loss = nll_loss + loss_dino
            loss += (self.codebook_weight * codebook_loss1.mean()) + (self.codebook_weight * codebook_loss2.mean())
            if entropy_loss1 is not None:
                loss += (self.entropy_loss_weight * entropy_loss1.mean()) + (self.entropy_loss_weight_q2 * entropy_loss2.mean())
            
            # loss += self.distill_loss_weight * distill_loss.mean()

            log = {
                f"{split}/total_loss": loss.detach().mean(),
                f"{split}/quant_loss1": codebook_loss1.detach().mean(),
                f"{split}/quant_loss2": codebook_loss2.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/p_loss": p_loss.detach().mean(),
                f"{split}/dino_loss": loss_dino.detach().mean(),
                # f"{split}/d_weight": d_weight,
                # f"{split}/disc_factor": torch.tensor(disc_factor),
                # f"{split}/g_loss": g_loss.detach().mean(),
            }
            if entropy_loss1 is not None:
                log[f"{split}/entropy_loss1"] = entropy_loss1.detach().mean()
                log[f"{split}/entropy_loss2"] = entropy_loss2.detach().mean()
                log[f"{split}/entropy_loss_weight"] = torch.tensor(self.entropy_loss_weight)
                log[f"{split}/entropy_loss_weight_q2"] = torch.tensor(self.entropy_loss_weight_q2)

            return loss, log