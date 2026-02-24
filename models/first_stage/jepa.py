import math
import random
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from util import instantiate_from_config

class Jepa(pl.LightningModule):
    """
    Predictive SSL Model

    context_encoder : masked i/p img
    target encoder : fully visible i/p img
    predictor : i/p context encoder representation at mask positions, pos embeds for those mask positions
    loss : l1 loss (predicted_masks, traget tokens at mask indices)

    """
    
    def __init__(
        self,
        ema_momentum,
        context_encoder_config,
        target_encoder_config,
        quantizer_config,
        decoder_config,
        loss_config,
        grad_acc_steps=1,
        cont_ratio_trainig= 0.0,
        ignore_keys=None,
        monitor=None,
        entropy_loss_weight_scheduler_config=None,
        min_lr_multiplier=0.1,
        only_decoder=False,
        scale_equivariance=None,
    ):
        super().__init__()

        ignore_keys = ignore_keys or []
        self.automatic_optimization = False
        self.grad_acc_steps = grad_acc_steps
        self.monitor = monitor
        # self.distill_model_type = distill_model_type
        self.cont_ratio_trainig = cont_ratio_trainig
        self.only_decoder=only_decoder
        self.min_lr_multiplier = min_lr_multiplier
        self.ema_momentum = ema_momentum
        
        self.gan_loss = None
        
        assert (not scale_equivariance) or len(scale_equivariance) == 2, "if defined, scale_equivariance should be a list of two lists"
        self.scale_equivariance = scale_equivariance

        # Instantiate core components
        self.context_encoder = instantiate_from_config(context_encoder_config)
        self.target_encoder = instantiate_from_config(target_encoder_config)
        self.quantize = instantiate_from_config(quantizer_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.loss = instantiate_from_config(loss_config)
        self.entropy_loss_weight_scheduler = instantiate_from_config(entropy_loss_weight_scheduler_config)

        # projection for feature comparison : [768 -> 384]
        self.quant_conv = nn.Conv2d(context_encoder_config.params["z_channels"], quantizer_config.params["e_dim"], 1)
        self.post_quant_conv = nn.Conv2d(quantizer_config.params["e_dim"], context_encoder_config.params["z_channels"], 1)

        self.encoder_normalize_embedding = context_encoder_config.params.get("normalize_embedding", False)
        self.quantizer_normalize_embedding = quantizer_config.params.get("normalize_embedding", False)

        # Image and patch size
        self.image_size = context_encoder_config.params["resolution"]
        self.patch_size = context_encoder_config.params["patch_size"]

        self.target_encoder.load_state_dict(self.context_encoder.state_dict())

        for p in self.target_encoder.parameters():
            p.requires_grad = False
    
    def get_input(self, batch):
        for k, v in batch.items():
            x = batch["images"]
            b, f, c, h, w = x.shape # [B, 1, 3, 256, 256]
            x = x.reshape(b, f*c, h, w)
        return x.float()

    def entropy_loss_weight_scheduling(self):
        self.loss.entropy_loss_weight = self.entropy_loss_weight_scheduler(self.global_step)

    def context_encode(self, x):
        h = self.context_encoder(x)
        h = self.quant_conv(h)
        
        if self.encoder_normalize_embedding:
            h = F.normalize(h, p=2, dim=1)

        ret = self.quantize(h)

        h = self.post_quant_conv(ret["quantized"])
        return ret, h
    
    def target_encode(self, x):
        h = self.target_encoder(x)
        return h
    
    def decode(self, quant):
        # distill_conv_out = self.post_quant_conv_distill(quant)
        rec = self.decoder(quant)
        return rec
        
    def forward(self, input):
        cb_losses, context = self.context_encode(input)
        with torch.no_grad():
            target = self.target_encode(input)
        rec = self.decode(context.detach())
        return rec, (cb_losses['quantization_loss'], cb_losses['entropy_loss']), context, target


    def l1_loss(self, context_feats, target_feats):

        target_feats = target_feats.detach()
        B, D, H, W = context_feats.shape

        context_feats = context_feats.reshape(B, D, H*W).permute(0, 2, 1)
        target_feats = target_feats.reshape(B, D, H*W).permute(0, 2, 1)

        context_feats = F.normalize(context_feats, p=2, dim=-1)
        target_feats = F.normalize(target_feats, p=2, dim=-1)

        cos_sim = F.cosine_similarity(context_feats, target_feats, dim=-1)
        
        return (1 - cos_sim).mean()
        # return F.l1_loss(context_feats, target_feats)

    @torch.no_grad()
    def update_target_encoder(self):
        context_params = dict(self.context_encoder.named_parameters())
        skipped_layers = 0
        updated_layers = 0
        prefix_to_strip = 'context_encoder.'

        for target_name, p_t in self.target_encoder.named_parameters():
            if target_name.startswith(prefix_to_strip):
                context_name = target_name[len(prefix_to_strip):]
            else:
                context_name = target_name

            if context_name in context_params:
                p_s = context_params[context_name]
                if p_s.shape == p_t.shape:
                    p_t.data = self.ema_momentum * p_t.data + (1 - self.ema_momentum) * p_s.data
                    updated_layers += 1
                else:
                    print(f"EMA Skipped: Shape mismatch for key '{context_name}' ({p_s.shape} vs {p_t.shape})")
                    skipped_layers += 1
            else:
                print(f"EMA Skipped: Parameter '{context_name}' not found in student encoder.")
                skipped_layers += 1

        print(f">> Updated {updated_layers} parameter tensors.")
        print(f"!! Skipped {skipped_layers} parameter tensors (due to missing key or shape mismatch).")

    def get_warmup_scheduler(self, optimizer, warmup_steps, min_lr_multiplier):
        min_lr = self.learning_rate * min_lr_multiplier
        total_steps = self.trainer.max_epochs * self.num_iters_per_epoch
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step/warmup_steps
            # After warmup_steps, we just return 1. This could be modified to implement your own schedule
            else:
                # progress = (step - warmup_steps) / (total_steps - warmup_steps)
                # cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                # decayed = (1 - min_lr) * cosine_decay + min_lr
                # return decayed
                return 1.0
              
        return LambdaLR(optimizer, lr_lambda)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.context_encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))
        
        scheduler_ae_warmup = self.get_warmup_scheduler(opt_ae, self.loss.warmup_steps, self.min_lr_multiplier)        

        return [opt_ae], [scheduler_ae_warmup]
    
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch)
        xrec, qloss, context, target  = self(x)

        distill_loss = None
        aeloss, log_dict_ae = self.loss(qloss, distill_loss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        l1_loss = self.l1_loss(context, target)
        self.log("val/l1_loss", l1_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        self.entropy_loss_weight_scheduling()
        self.log("train/enropy_loss_weight", self.loss.entropy_loss_weight, 
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        opt_ae = self.optimizers()
        scheduler_ae_warmup = self.lr_schedulers()
        
        x = self.get_input(batch)
        
        xrec, qloss, context, target  = self(x)

        distill_loss = None

        optimizer_idx = 0
        aeloss, log_dict_ae = self.loss(qloss, distill_loss, x, xrec, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        l1_loss = self.l1_loss(context, target)
        self.log("train/l1_loss", l1_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        aeloss = aeloss + l1_loss
        aeloss = aeloss / self.grad_acc_steps
        self.manual_backward(aeloss) 
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_ae.step()
            opt_ae.zero_grad()
            scheduler_ae_warmup.step()
            self.update_target_encoder()

    def get_last_layer(self):
        try:
            return self.decoder.conv_out.weight
        except:
            return None
        
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch)
        x = x.to(self.device)
        xrec, _, _, _ = self(x)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

class Jepa_1d(Jepa):    
    def __init__(
        self,
        ema_momentum,
        context_encoder_config,
        target_encoder_config,
        quantizer_config,
        decoder_config,
        loss_config,
        grad_acc_steps=1,
        cont_ratio_trainig= 0.0,
        ignore_keys=None,
        monitor=None,
        entropy_loss_weight_scheduler_config=None,
        min_lr_multiplier=0.1,
        only_decoder=False,
        scale_equivariance=None,
    ):
        super().__init__(ema_momentum, context_encoder_config, target_encoder_config, quantizer_config, decoder_config, 
                         loss_config, grad_acc_steps, cont_ratio_trainig, ignore_keys, monitor, entropy_loss_weight_scheduler_config,
                         min_lr_multiplier, only_decoder, scale_equivariance)


        # Instantiate core components
        self.context_encoder = instantiate_from_config(context_encoder_config)
        self.target_encoder = instantiate_from_config(target_encoder_config)
        self.quantize = instantiate_from_config(quantizer_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.loss = instantiate_from_config(loss_config)
        self.entropy_loss_weight_scheduler = instantiate_from_config(entropy_loss_weight_scheduler_config)

        # Convolutional layers for quantization
        self.quant_conv = nn.Linear(context_encoder_config.params["z_channels"], quantizer_config.params["e_dim"])
        self.post_quant_conv = nn.Linear(quantizer_config.params["e_dim"], context_encoder_config.params["z_channels"])
    
    def context_encode(self, x):
        h = self.context_encoder(x)
        h = self.quant_conv(h)
        h = h.permute(0, 2, 1)
        if self.encoder_normalize_embedding:
            h = F.normalize(h, p=2, dim=1)
        h = h.unsqueeze(2)                  # [B, C, 1, N]
        ret = self.quantize(h)
        ret["quantized"] = ret["quantized"].squeeze(2).permute(0, 2, 1)
        ret["quantized"] = self.post_quant_conv(ret["quantized"])
        return ret
    
    def target_encode(self, x):
        h = self.target_encoder(x) # [B, C, H, W]
        B, D, H, W = h.shape
        h = h.reshape(B, D, H*W).permute(0, 2, 1)
        return h
    
    def decode(self, quant):
        # distill_conv_out = self.post_quant_conv_distill(quant)
        rec = self.decoder(quant)
        return rec
        
    def forward(self, input):
        context = self.context_encode(input)
        quant_tokens = context["quantized"]
        with torch.no_grad():
            target = self.target_encode(input)
        rec = self.decode(quant_tokens.detach())
        return rec, (context['quantization_loss'], context['entropy_loss']), quant_tokens, target

    
    def l1_loss(self, context_feats, target_feats):

        target_feats = target_feats.detach()

        context_feats = F.normalize(context_feats, p=2, dim=-1)
        target_feats = F.normalize(target_feats, p=2, dim=-1)

        cos_sim = F.cosine_similarity(context_feats, target_feats, dim=-1)
        
        return (1 - cos_sim).mean()
        # return F.l1_loss(context_feats, target_feats)

    
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch)
        xrec, qloss, context, target  = self(x)

        distill_loss = None
        aeloss, log_dict_ae = self.loss(qloss, distill_loss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        l1_loss = self.l1_loss(context, target)
        self.log("val/l1_loss", l1_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        

    def training_step(self, batch, batch_idx):
        self.entropy_loss_weight_scheduling()
        self.log("train/enropy_loss_weight", self.loss.entropy_loss_weight, 
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        opt_ae = self.optimizers()
        scheduler_ae_warmup = self.lr_schedulers()
        
        x = self.get_input(batch)
        
        xrec, qloss, context, target  = self(x)

        distill_loss = None

        optimizer_idx = 0
        aeloss, log_dict_ae = self.loss(qloss, distill_loss, x, xrec, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        l1_loss = self.l1_loss(context, target)
        self.log("train/l1_loss", l1_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        aeloss = aeloss + l1_loss
        aeloss = aeloss / self.grad_acc_steps
        self.manual_backward(aeloss) 
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_ae.step()
            opt_ae.zero_grad()
            scheduler_ae_warmup.step()
            self.update_target_encoder()

    def get_last_layer(self):
        try:
            return self.decoder.conv_out.weight
        except:
            return None
        
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch)
        x = x.to(self.device)
        xrec, _, _, _ = self(x)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

class Jepa_weak(Jepa):
    def __init__(
        self,
        ema_momentum,
        context_encoder_config,
        target_encoder_config,
        quantizer_config,
        decoder_config,
        loss_config,
        grad_acc_steps=1,
        cont_ratio_trainig= 0.0,
        ignore_keys=None,
        monitor=None,
        entropy_loss_weight_scheduler_config=None,
        min_lr_multiplier=0.1,
        only_decoder=False,
        scale_equivariance=None,
    ):
        super().__init__(ema_momentum, context_encoder_config, target_encoder_config, quantizer_config, decoder_config, 
                         loss_config, grad_acc_steps, cont_ratio_trainig, ignore_keys, monitor, entropy_loss_weight_scheduler_config,
                         min_lr_multiplier, only_decoder, scale_equivariance)
        
    def forward(self, input):
        alpha = 0.02
        cb_losses, context = self.context_encode(input)
        with torch.no_grad():
            target = self.target_encode(input)
        context_enc = context * alpha + context.detach() * (1 - alpha) 
        rec = self.decode(context_enc)
        return rec, (cb_losses['quantization_loss'], cb_losses['entropy_loss']), context, target
