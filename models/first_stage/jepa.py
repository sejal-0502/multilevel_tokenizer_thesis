import math
import random
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from util import instantiate_from_config
      

###############################################################################
# Model : EMA-Supervision in the decoder feature space
###############################################################################      

class Jepa_dualbranch_decoderbased(pl.LightningModule):
    def __init__(
        self,
        ema_momentum,
        mask_ratio,
        encoder_config,
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
        self.mask_ratio = mask_ratio
        
        self.gan_loss = None
        
        assert (not scale_equivariance) or len(scale_equivariance) == 2, "if defined, scale_equivariance should be a list of two lists"
        self.scale_equivariance = scale_equivariance

        # Instantiate core components
        self.context_encoder = instantiate_from_config(encoder_config)
        self.target_encoder = instantiate_from_config(encoder_config)
        self.quantize = instantiate_from_config(quantizer_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.loss = instantiate_from_config(loss_config)
        self.entropy_loss_weight_scheduler = instantiate_from_config(entropy_loss_weight_scheduler_config)

        # Convolutional layers for quantization
        self.quant_conv = nn.Conv2d(encoder_config.params["z_channels"], quantizer_config.params["e_dim"], 1)
        self.post_quant_conv = nn.Conv2d(quantizer_config.params["e_dim"], decoder_config.params["z_channels"], 1)

        self.encoder_normalize_embedding = encoder_config.params.get("normalize_embedding", False)
        self.quantizer_normalize_embedding = quantizer_config.params.get("normalize_embedding", False)

        # Image and patch size
        self.image_size = encoder_config.params["resolution"]
        self.patch_size = encoder_config.params["patch_size"]

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
        h = self.context_encoder(x, self.mask_ratio, mode="student")
        h = self.quant_conv(h)
        if self.encoder_normalize_embedding:
            h = F.normalize(h, p=2, dim=1)

        ret_spatial = self.quantize(h)
        ret_spatial["continuous"] = h
        
        return ret_spatial
    
    def target_encode(self, x):
        h = self.target_encoder(x, self.mask_ratio, mode="teacher")
        B, D, H, W = h.shape
        h = h.reshape(B, D, H*W).permute(0, 2, 1)
        h = F.normalize(h, p=2, dim=-1)
        return h
    
    def decode(self, quant):
        quant2 = self.post_quant_conv(quant)
        self.decoder.give_pre_end = True
        rec, rec_target = self.decoder(quant2)
        self.decoder.give_pre_end = False
        rec_target = F.normalize(rec_target, p=2, dim=-1)
        return rec, rec_target

    def forward(self, input):
        encoded_spatial = self.context_encode(input)
        with torch.no_grad():
            target = self.target_encode(input)
        rec_recon, rec_target = self.decode(encoded_spatial["quantized"])
        return rec_recon, rec_target, (encoded_spatial['quantization_loss'], encoded_spatial['entropy_loss']), target
    
    @torch.no_grad()
    def update_target_encoder(self):
        m = self.ema_momentum

        for p_s, p_t in zip(self.context_encoder.parameters(),
                            self.target_encoder.parameters()):
            p_t.data.mul_(m).add_((1 - m) * p_s.data)

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
        xrec, dec_target, qloss, target  = self(x)

        distill_loss = None
        aeloss, log_dict_ae = self.loss(qloss, distill_loss, x, xrec, target, dec_target, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        jepa_loss = log_dict_ae["val/jepa_loss"]
        self.log("val/jepa_loss", jepa_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        self.entropy_loss_weight_scheduling()
        self.log("train/enropy_loss_weight", self.loss.entropy_loss_weight, 
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        opt_ae = self.optimizers()
        scheduler_ae_warmup = self.lr_schedulers()
        
        x = self.get_input(batch)
        
        xrec, dec_target, qloss, target  = self(x)

        distill_loss = None

        optimizer_idx = 0
        aeloss, log_dict_ae = self.loss(qloss, distill_loss, x, xrec, target, dec_target, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        
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
    
class Jepa_bootleg(Jepa_dualbranch_decoderbased):
    def __init__(
        self,
        ema_momentum,
        mask_ratio,
        encoder_config,
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
        super().__init__(ema_momentum,mask_ratio,encoder_config,quantizer_config,decoder_config,loss_config,grad_acc_steps,
                         cont_ratio_trainig,ignore_keys,monitor,entropy_loss_weight_scheduler_config,min_lr_multiplier,
                         only_decoder,scale_equivariance)
        
    
        in_dim = 768
        hidden_dim = 64
        out_dim = 768
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=False),
                                 nn.LayerNorm(hidden_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, out_dim))


    def context_encode(self, x):
        h = self.context_encoder(x, self.mask_ratio, mode="student")
        ret_target = h
        h = self.quant_conv(h)
        if self.encoder_normalize_embedding:
            h = F.normalize(h, p=2, dim=1)

        ret_spatial = self.quantize(h)
        
        return ret_spatial, ret_target
    
    def target_encode(self, x):
        h = self.target_encoder(x, self.mask_ratio, mode="teacher")
        B, D, H, W = h.shape
        h = h.reshape(B, D, H*W).permute(0, 2, 1)
        h = F.normalize(h, p=2, dim=-1)
        return h

    def predictor(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H*W).permute(0, 2, 1)
        x = self.net(x)
        x = F.normalize(x, p=2, dim=-1)
        return x
    
    def decode(self, quant):
        # distill_conv_out = self.post_quant_conv_distill(quant)
        quant2 = self.post_quant_conv(quant)
        rec = self.decoder(quant2)
        return rec

    def forward(self, input):
        encoded_spatial, encoded_target = self.context_encode(input)
        encoded_target = self.predictor(encoded_target)
        with torch.no_grad():
            target = self.target_encode(input)
        rec_recon = self.decode(encoded_spatial["quantized"])
        return rec_recon, encoded_target, (encoded_spatial['quantization_loss'], encoded_spatial['entropy_loss']), target
    
    @torch.no_grad()
    def update_target_encoder(self):
        m = self.ema_momentum

        for p_s, p_t in zip(self.context_encoder.parameters(),
                            self.target_encoder.parameters()):
            p_t.data.mul_(m).add_((1 - m) * p_s.data)

    
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch)
        xrec, encoded_target, qloss, target  = self(x)

        distill_loss = None
        aeloss, log_dict_ae = self.loss(qloss, distill_loss, x, xrec, target, encoded_target, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        jepa_loss = log_dict_ae["val/jepa_loss"]
        self.log("val/jepa_loss", jepa_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        self.entropy_loss_weight_scheduling()
        self.log("train/enropy_loss_weight", self.loss.entropy_loss_weight, 
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        opt_ae = self.optimizers()
        scheduler_ae_warmup = self.lr_schedulers()
        
        x = self.get_input(batch)
        
        xrec, encoded_target, qloss, target  = self(x)

        distill_loss = None

        optimizer_idx = 0
        aeloss, log_dict_ae = self.loss(qloss, distill_loss, x, xrec, target, encoded_target, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        
        aeloss = aeloss / self.grad_acc_steps
        self.manual_backward(aeloss) 
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_ae.step()
            opt_ae.zero_grad()
            scheduler_ae_warmup.step()
            self.update_target_encoder()