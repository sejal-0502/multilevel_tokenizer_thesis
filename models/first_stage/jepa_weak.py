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
        self.quantizer = instantiate_from_config(quantizer_config)
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
        ret = self.quantizer(h)
        h = self.post_quant_conv(ret["quantized"])
        # ret["continuous"] = h
        return ret, h
    
    def target_encode(self, x):
        h = self.target_encoder(x)
        return h
    
    def decode(self, quant):
        # distill_conv_out = self.post_quant_conv_distill(quant)
        rec = self.decoder(quant)
        return rec
        
    def forward(self, input):
        alpha = 0.1
        cb_losses, context = self.context_encode(input)
        with torch.no_grad():
            target = self.target_encode(input)
        context_enc = context * alpha + context.detach() * (1 - alpha) 
        rec = self.decode(context_enc)
        return rec, (cb_losses['quantization_loss'], cb_losses['entropy_loss']), context, target
    
    def l1_loss(self, context_feats, target_feats):
        return F.l1_loss(context_feats, target_feats)
    
    def grad_norm(self, params):
        return sum(p.grad.norm().item()**2 for p in params if p.grad is not None) ** 0.5

    @torch.no_grad()
    def update_target_encoder(self):
        momentum = self.ema_momentum
        for param_q, param_k in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data


    def get_warmup_scheduler(self, optimizer, warmup_steps, min_lr_multiplier):
        min_lr = self.learning_rate * min_lr_multiplier
        total_steps = self.trainer.max_epochs * self.num_iters_per_epoch
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step/warmup_steps
            # After warmup_steps, we just return 1. This could be modified to implement your own schedule
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                decayed = (1 - min_lr) * cosine_decay + min_lr
                return decayed
              
        return LambdaLR(optimizer, lr_lambda)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.context_encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantizer.parameters())+
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

        # Encoder gradient (includes both L1 + alpha-scaled recon)
        encoder_grad_norm = self.grad_norm(self.context_encoder.parameters())
        self.log("train/encoder_grad_norm", encoder_grad_norm, prog_bar=True, logger=True)

        # Decoder gradient (full recon)
        decoder_grad_norm = self.grad_norm(self.decoder.parameters())
        self.log("train/decoder_grad_norm", decoder_grad_norm, prog_bar=True, logger=True)

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
    
class Jepa_aplhaScheduler(Jepa):
    """
    Alpa scheduled version of weak jepa
    """
    def __init__(
        self,
        ema_momentum,
        alpha_start,
        alpha_end,
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
        super().__init__(ema_momentum, context_encoder_config, target_encoder_config, quantizer_config, 
                         decoder_config, loss_config, grad_acc_steps, cont_ratio_trainig, ignore_keys,
                         monitor, entropy_loss_weight_scheduler_config, min_lr_multiplier, 
                         only_decoder, scale_equivariance,)
        
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha_ramp_step = None
        self.alpha_ramp_rate = None

    def on_train_start(self):    
        step_per_epoch = len(self.trainer.train_dataloader)
        optimizer_step_per_epoch = step_per_epoch // self.grad_acc_steps
        self.alpha_ramp_step = optimizer_step_per_epoch * 4                 # ramp over 4 epochs
        self.alpha_ramp_rate = (self.alpha_end - self.alpha_start) / self.alpha_ramp_step
        return self.alpha_ramp_rate

    def get_alpha(self):
        """Return current alpha based on global_step (optimizer step)."""
        if self.alpha_ramp_rate is None or self.alpha_start is None:
            return self.alpha_start if self.alpha_start is not None else 0.0
        
        alpha = self.alpha_start + self.global_step * self.alpha_ramp_rate
        return min(alpha, self.alpha_end)
        
    def forward(self, input):
        alpha = self.get_alpha()
        cb_losses, context = self.context_encode(input)
        with torch.no_grad():
            target = self.target_encode(input)
        context_enc = context * alpha + context.detach() * (1 - alpha) 
        rec = self.decode(context_enc)

        self.log("train/alpha", alpha, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
    
        return rec, (cb_losses['quantization_loss'], cb_losses['entropy_loss']), context, target