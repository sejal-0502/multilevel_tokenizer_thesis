import math
import random
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.pos_embed import resample_abs_pos_embed
from torch.optim.lr_scheduler import LambdaLR

from util import instantiate_from_config

class VQModel(pl.LightningModule):
    """
    VQGAN model: vector-quantized autoencoder with adversarial training.
    """
    
    def __init__(
        self,
        encoder_config,
        decoder_config,
        quantizer_config,
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
        self.norm_pix_loss = False
        self.grad_acc_steps = grad_acc_steps
        self.monitor = monitor
        self.cont_ratio_trainig = cont_ratio_trainig
        self.only_decoder=only_decoder
        self.min_lr_multiplier = min_lr_multiplier
        
        assert (not scale_equivariance) or len(scale_equivariance) == 2, "if defined, scale_equivariance should be a list of two lists"
        self.scale_equivariance = scale_equivariance

        # Decoder uses encoder params if none provided
        if not hasattr(decoder_config, "params"):
            decoder_config.params = encoder_config.params

        # Instantiate core components
        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.quantize = instantiate_from_config(quantizer_config)
        self.loss = instantiate_from_config(loss_config)
        self.entropy_loss_weight_scheduler = instantiate_from_config(entropy_loss_weight_scheduler_config)

        # Convolutional layers for quantization
        self.quant_conv = nn.Conv2d(encoder_config.params["z_channels"], quantizer_config.params["e_dim"], 1)
        self.post_quant_conv = nn.Conv2d(quantizer_config.params["e_dim"], decoder_config.params["z_channels"], 1)

        self.encoder_normalize_embedding = encoder_config.params.get("normalize_embedding", False)
        self.quantizer_normalize_embedding = quantizer_config.params.get("normalize_embedding", False)

        self.if_distill_loss = False if loss_config.params.get('distill_loss_weight', 0.0) == 0.0 else True
        
        # Image and patch size
        self.image_size = encoder_config.params["resolution"]
        self.patch_size = encoder_config.params["patch_size"]
    
    def get_input(self, batch):
        for k, v in batch.items():
            x = batch["images"]
            b, f, c, h, w = x.shape # [B, 1, 3, 256, 256]
            x = x.reshape(b, f*c, h, w)
        return x.float()

    def entropy_loss_weight_scheduling(self):
        self.loss.entropy_loss_weight = self.entropy_loss_weight_scheduler(self.global_step)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        if self.encoder_normalize_embedding:
            h = F.normalize(h, p=2, dim=1)
        ret = self.quantize(h)
        # ret["continuous"] = h
        return ret
        
    def decode(self, quant):
        # distill_conv_out = self.post_quant_conv_distill(quant)
        quant2 = self.post_quant_conv(quant)
        rec = self.decoder(quant2)
        return rec
    
    def forward(self, input):
        encoded = self.encode(input)
        rec = self.decode(encoded["quantized"])
        return rec, (encoded['quantization_loss'], encoded['entropy_loss'])
    
    def get_warmup_scheduler(self, optimizer, warmup_steps, min_lr_multiplier):
        min_lr = self.learning_rate * min_lr_multiplier
        total_steps = self.trainer.max_epochs * self.num_iters_per_epoch
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step/warmup_steps
            # After warmup_steps, we just return 1. This could be modified to implement your own schedule
            else:
                return 1.0       
        
        return LambdaLR(optimizer, lr_lambda)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))
        
        scheduler_ae_warmup = self.get_warmup_scheduler(opt_ae, self.loss.warmup_steps, self.min_lr_multiplier)        

        return [opt_ae], [scheduler_ae_warmup]
    
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch)
        xrec, qloss = self(x)

        distill_loss = None
        aeloss, log_dict_ae = self.loss(qloss, distill_loss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        self.entropy_loss_weight_scheduling()
        self.log("train/enropy_loss_weight", self.loss.entropy_loss_weight, 
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        opt_ae = self.optimizers()
        scheduler_ae_warmup = self.lr_schedulers()
        
        x = self.get_input(batch)
        
        xrec, qloss = self(x)

        distill_loss = None

        optimizer_idx = 0
        aeloss, log_dict_ae = self.loss(qloss, distill_loss, x, xrec, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        aeloss = aeloss / self.grad_acc_steps
        self.manual_backward(aeloss) 
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_ae.step()
            opt_ae.zero_grad()
            scheduler_ae_warmup.step()

    def get_last_layer(self):
        try:
            return self.decoder.conv_out.weight
        except:
            return None
        
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch)
        x = x.to(self.device)
        xrec, _ = self(x)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

class VQModel_multiframes(pl.LightningModule):
    """
    VQGAN model: vector-quantized autoencoder with adversarial training.
    """
    
    def __init__(
        self,
        encoder_config,
        decoder_config,
        quantizer_config,
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
        self.norm_pix_loss = False
        self.grad_acc_steps = grad_acc_steps
        self.monitor = monitor
        self.cont_ratio_trainig = cont_ratio_trainig
        self.only_decoder=only_decoder
        self.min_lr_multiplier = min_lr_multiplier
        
        assert (not scale_equivariance) or len(scale_equivariance) == 2, "if defined, scale_equivariance should be a list of two lists"
        self.scale_equivariance = scale_equivariance

        # Decoder uses encoder params if none provided
        if not hasattr(decoder_config, "params"):
            decoder_config.params = encoder_config.params

        # Instantiate core components
        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.quantize = instantiate_from_config(quantizer_config)
        self.loss = instantiate_from_config(loss_config)
        self.entropy_loss_weight_scheduler = instantiate_from_config(entropy_loss_weight_scheduler_config)

        # Convolutional layers for quantization
        self.quant_conv = nn.Conv2d(encoder_config.params["z_channels"], quantizer_config.params["e_dim"], 1)
        self.post_quant_conv = nn.Conv2d(quantizer_config.params["e_dim"], decoder_config.params["z_channels"], 1)

        self.encoder_normalize_embedding = encoder_config.params.get("normalize_embedding", False)
        self.quantizer_normalize_embedding = quantizer_config.params.get("normalize_embedding", False)

        self.if_distill_loss = False if loss_config.params.get('distill_loss_weight', 0.0) == 0.0 else True
        
        # Image and patch size
        self.image_size = encoder_config.params["resolution"]
        self.patch_size = encoder_config.params["patch_size"]
    
    def get_input(self, batch):
        for k, v in batch.items():
            x = batch["images"]
            b, f, c, h, w = x.shape # [B, 2, 3, 256, 256]
            
            x1 = x[:, 0] # [B, 3, 256, 256]
            x2 = x[:, 1]
            
        return x1.float(), x2.float()

    def entropy_loss_weight_scheduling(self):
        self.loss.entropy_loss_weight = self.entropy_loss_weight_scheduler(self.global_step)

    def encode(self, x1, x2):
        h = self.encoder(x1, x2)
        h = self.quant_conv(h)
        if self.encoder_normalize_embedding:
            h = F.normalize(h, p=2, dim=1)
        ret = self.quantize(h)
        # ret["continuous"] = h
        return ret
        
    def decode(self, quant):
        # distill_conv_out = self.post_quant_conv_distill(quant)
        quant2 = self.post_quant_conv(quant)
        rec = self.decoder(quant2)
        return rec
    
    def forward(self, input1, input2):
        encoded = self.encode(input1, input2)
        rec = self.decode(encoded["quantized"])
        return rec, (encoded['quantization_loss'], encoded['entropy_loss'])
    
    def get_warmup_scheduler(self, optimizer, warmup_steps, min_lr_multiplier):
        min_lr = self.learning_rate * min_lr_multiplier
        total_steps = self.trainer.max_epochs * self.num_iters_per_epoch
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step/warmup_steps
            # After warmup_steps, we just return 1. This could be modified to implement your own schedule
            else:
                return 1.0       
        
        return LambdaLR(optimizer, lr_lambda)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))
        
        scheduler_ae_warmup = self.get_warmup_scheduler(opt_ae, self.loss.warmup_steps, self.min_lr_multiplier)        

        return [opt_ae], [scheduler_ae_warmup]
    
    def validation_step(self, batch, batch_idx):
        x1, x2 = self.get_input(batch)
        xrec, qloss = self(x1, x2)

        distill_loss = None
        aeloss, log_dict_ae = self.loss(qloss, distill_loss, x2, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        self.entropy_loss_weight_scheduling()
        self.log("train/enropy_loss_weight", self.loss.entropy_loss_weight, 
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        opt_ae = self.optimizers()
        scheduler_ae_warmup = self.lr_schedulers()
        
        x1, x2 = self.get_input(batch)
        
        xrec, qloss = self(x1, x2)

        distill_loss = None

        optimizer_idx = 0
        aeloss, log_dict_ae = self.loss(qloss, distill_loss, x2, xrec, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        aeloss = aeloss / self.grad_acc_steps
        self.manual_backward(aeloss) 
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_ae.step()
            opt_ae.zero_grad()
            scheduler_ae_warmup.step()

    def get_last_layer(self):
        try:
            return self.decoder.conv_out.weight
        except:
            return None
        
    def log_images(self, batch, **kwargs):
        log = dict()
        x1, x2 = self.get_input(batch)
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        xrec, _ = self(x1, x2)
        log["input_t"] = x1
        log["input_t+1"] = x2
        log["reconstruction_t+1"] = xrec
        return log

class VQModel_multiframes_1d(VQModel_multiframes):
    """
    VQGAN model: vector-quantized autoencoder with adversarial training.
    """
    def __init__(
        self,
        encoder_config,
        decoder_config,
        quantizer_config,
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
        super().__init__(encoder_config, decoder_config, quantizer_config, loss_config, grad_acc_steps,
                         cont_ratio_trainig, ignore_keys, monitor, entropy_loss_weight_scheduler_config,
                         min_lr_multiplier, only_decoder, scale_equivariance,)       

        # Convolutional layers for quantization
        self.quant_conv = nn.Linear(encoder_config.params["z_channels"], quantizer_config.params["e_dim"])
        self.post_quant_conv = nn.Linear(quantizer_config.params["e_dim"], decoder_config.params["z_channels"])
    
    def encode(self, x1, x2):
        h = self.encoder(x1, x2)
        h = self.quant_conv(h)
        h = h.permute(0, 2, 1)
        if self.encoder_normalize_embedding:
            h = F.normalize(h, p=2, dim=1)
        h = h.unsqueeze(2)                  # [B, C, 1, N]
        ret = self.quantize(h)
        ret["quantized"] = ret["quantized"].squeeze(2).permute(0, 2, 1) # [B, N, C]
        return ret
    
    def decode(self, quant):
        # distill_conv_out = self.post_quant_conv_distill(quant)
        quant2 = self.post_quant_conv(quant)
        rec = self.decoder(quant2)
        return rec


class VQModelIF(VQModel):
    """
    VQGAN model with token factorization (IF: ImageFolder)
    """
    def __init__(self, 
                 encoder_config,
                 decoder_config,
                 quantizer_config,
                 loss_config,
                 grad_acc_steps=1,
                 cont_ratio_trainig= 0.0,
                 ignore_keys=[],
                 monitor=None,
                 entropy_loss_weight_scheduler_config=None,
                 distill_model_type='VIT_DINOv2', # 'VIT_DINO' or 'CNN' or VIT_DINOv2, VIT_DINOv2_large_reg4, SAM_VIT
                 min_lr_multiplier=0.1,
                 only_decoder=False,
                 scale_equivariance=[]
                 ):
        super().__init__(encoder_config, decoder_config, quantizer_config, loss_config, 
                         grad_acc_steps, cont_ratio_trainig, ignore_keys, 
                         monitor, 
                         entropy_loss_weight_scheduler_config, 
                         distill_model_type, min_lr_multiplier, only_decoder, scale_equivariance)
    
        self.encoder2 = instantiate_from_config(encoder_config)
        self.post_quant_conv = torch.nn.Conv2d(quantizer_config.params['e_dim']*2, decoder_config.params["z_channels"], 1)
        self.quant_conv2 = torch.nn.Conv2d(encoder_config.params["z_channels"], quantizer_config.params['e_dim'], 1)
        self.quantize2 = instantiate_from_config(quantizer_config)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.encoder2.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quantize2.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.quant_conv2.parameters())+
                                  list(self.post_quant_conv.parameters())+
                                  list(self.post_quant_conv_distill.parameters()),
                                  lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))
        
        scheduler_ae_warmup = self.get_warmup_scheduler(opt_ae, self.loss.warmup_steps, self.min_lr_multiplier)
        scheduler_disc_warmup = self.get_warmup_scheduler(opt_disc, self.loss.warmup_steps, self.min_lr_multiplier)
        

        return [opt_ae, opt_disc], [scheduler_ae_warmup, scheduler_disc_warmup]

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        if self.encoder_normalize_embedding:
            h = F.normalize(h, p=2, dim=1)

        h2 = self.encoder2(x)
        h2 = self.quant_conv2(h2)
        if self.encoder_normalize_embedding:
            h2 = F.normalize(h2, p=2, dim=1)

        quant = self.quantize(h)
        quant2 = self.quantize2(h2)
        
        quant_loss = quant['quantization_loss'] + quant2['quantization_loss']
        entropy_loss = quant['entropy_loss'] + quant2['entropy_loss'] if quant['entropy_loss'] is not None and quant2['entropy_loss'] is not None else None
        
        ret = {
            "quantized": (quant["quantized"], quant2["quantized"]),
            "quantization_loss": quant_loss,
            "entropy_loss": entropy_loss,
            "indices": (quant["indices"], quant2["indices"]),
            "continuous": (h, h2)
        }
        return ret
    

    def decode(self, quant):
        if isinstance(quant, tuple):
            quant_rec = quant[0]
            quant_sem = quant[1]
        else:
            print('Error: quant should be a tuple')
        distill_conv_out = self.post_quant_conv_distill(quant_sem)
        quant_cat = torch.cat((quant_rec, quant_sem), dim=1)
        quant = self.post_quant_conv(quant_cat)
        return self.decoder(quant), distill_conv_out
    
    def decode_code(self, code_b):
        code_b_rec, code_b_sem = code_b
        quant_b_rec = self.quantize.get_codebook_entry(code_b_rec, (-1, code_b_rec.size(1), code_b_rec.size(2), self.quantize.e_dim))
        quant_b_sem = self.quantize2.get_codebook_entry(code_b_sem, (-1, code_b_sem.size(1), code_b_sem.size(2), self.quantize.e_dim))
        quant_b = (quant_b_rec, quant_b_sem)
        dec = self.decode(quant_b)
        return dec
    
    def forward_se(self, input):
        random_scale = [random.choice(self.scale_equivariance[0]), random.choice(self.scale_equivariance[1])]
        downscale_factor = [1/random_scale[0], 1/random_scale[1]]
        encoded = self.encode(input)
        quantized = encoded["quantized"]
        continuous = encoded["continuous"]
        if torch.rand(1) > self.cont_ratio_trainig:
            dec, distill_conv_out = self.decode(quantized)
            quant_se = F.interpolate(quantized[0], scale_factor=downscale_factor, mode='bilinear', align_corners=False), \
                       F.interpolate(quantized[1], scale_factor=downscale_factor, mode='bilinear', align_corners=False)
            dec_se = self.decode(quant_se)[0]
        else:
            dec, distill_conv_out = self.decode(continuous)
            latents_se =  F.interpolate(continuous[0], scale_factor=downscale_factor, mode='bilinear', align_corners=False), \
                          F.interpolate(continuous[1], scale_factor=downscale_factor, mode='bilinear', align_corners=False)
            dec_se = self.decode(latents_se)[0]

        input_se = F.interpolate(input, scale_factor=downscale_factor, mode='bilinear', align_corners=False)
        decs = [dec, dec_se]
        inputs = [input, input_se]
        return inputs, decs, (encoded["quantization_loss"], encoded["entropy_loss"]), distill_conv_out