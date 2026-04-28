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

######################################################################################
# Models : MAE, Temporal MAE, Distillation (DINO, Depth, RAFT), Temporal Compression
######################################################################################

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
        ret["continuous"] = h
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
    
class VQModel_finetune(VQModel):
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
        super().__init__(encoder_config,decoder_config,quantizer_config,loss_config,grad_acc_steps,cont_ratio_trainig,
                         ignore_keys,monitor,entropy_loss_weight_scheduler_config,min_lr_multiplier,only_decoder,
                         scale_equivariance)
        
        # Ensure only decoder-related modules are in training mode
        if self.only_decoder:
            # Freeze encoder side
            for module in [self.encoder, self.quantize, self.quant_conv]:
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False

        if self.only_decoder:
            self.encoder.eval()
            self.quantize.eval()
            self.quant_conv.eval()
    
    def forward(self, input):
        if self.only_decoder:
            with torch.no_grad():
                encoded = self.encode(input)
            quant = encoded["quantized"]
            rec = self.decode(quant)
            return rec, None
        else:
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
        if self.only_decoder:
            params = list(self.decoder.parameters()) + \
                list(self.post_quant_conv.parameters())
        else:
            params = list(self.encoder.parameters()) + \
                    list(self.decoder.parameters()) + \
                    list(self.quantize.parameters()) + \
                    list(self.quant_conv.parameters())

        opt_ae = torch.optim.Adam(
            params,
            lr=lr,
            betas=(self.loss.beta_1, self.loss.beta_2)
        )

        scheduler_ae_warmup = self.get_warmup_scheduler(
            opt_ae, self.loss.warmup_steps, self.min_lr_multiplier
        )

        return [opt_ae], [scheduler_ae_warmup]
    
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch)
        xrec, qloss = self(x)

        distill_loss = None
        qloss = torch.tensor(0.0, device=x.device)

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
        qloss = torch.tensor(0.0, device=x.device)

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
    
    def save_decoder(self, path):
        torch.save(self.decoder.state_dict(), path)

    
class VQModel_images(VQModel):
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
        super().__init__(encoder_config,decoder_config,quantizer_config,loss_config,grad_acc_steps,cont_ratio_trainig,
                         ignore_keys,monitor,entropy_loss_weight_scheduler_config,min_lr_multiplier,only_decoder,
                         scale_equivariance)

      
    def get_input(self, batch):
        for k in batch.items():
            x = batch["image"]
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()


class VQModel_1frame_motion(VQModel):
    """
    VAE motion model
    """
    def __init__(
        self,
        encoder_config,
        decoder_config,
        decoder_motion_config,
        quantizer_config,
        motion_config,
        loss_config,
        grad_acc_steps=1,
        cont_ratio_trainig= 0.0,
        ignore_keys=None,
        monitor=None,
        entropy_loss_weight_scheduler_config=None,
        entropy_loss_weight_scheduler_config_q2=None,
        min_lr_multiplier=0.1,
        only_decoder=False,
        scale_equivariance=None,
    ):
        super().__init__(encoder_config, decoder_config, quantizer_config, loss_config, grad_acc_steps,
                         cont_ratio_trainig, ignore_keys, monitor, entropy_loss_weight_scheduler_config,
                         min_lr_multiplier, only_decoder, scale_equivariance,)       

        self.quantize2 = instantiate_from_config(quantizer_config)  # motion quantizer
        self.decoder2 = instantiate_from_config(decoder_motion_config)     # motion decoder
        self.motion = instantiate_from_config(motion_config)
        self.entropy_loss_weight_scheduler_q2 = instantiate_from_config(entropy_loss_weight_scheduler_config_q2)
        
        # Convolutional layers for quantization
        self.quant_conv = nn.Conv2d(encoder_config.params["z_channels"], quantizer_config.params["e_dim"], 1)
        self.post_quant_conv = nn.Conv2d(quantizer_config.params["e_dim"], decoder_config.params["z_channels"], 1)

        self.quant_conv2 = nn.Conv2d(encoder_config.params["z_channels"], quantizer_config.params["e_dim"], 1)
        self.post_quant_conv2 = nn.Conv2d(quantizer_config.params["e_dim"], decoder_motion_config.params["z_channels"], 1)

    def entropy_loss_weight_scheduling_q2(self):
        self.loss.entropy_loss_weight_q2 = self.entropy_loss_weight_scheduler_q2(self.global_step)

    def get_input(self, batch):
        for k, v in batch.items():
            x = batch["images"]
            b, f, c, h, w = x.shape # [B, 2, 3, 256, 256]
            
            x1 = x[:, 0] # [B, 3, 256, 256]
            x2 = x[:, 1]
            
        return x1.float(), x2.float()
    
    def gt_motion(self, x1, x2):
        h = self.motion(x1, x2)         # [B, N, 1]
        return h
    
    def encode(self, x):
        h = self.encoder(x)
        h1 = self.quant_conv(h)
        h2 = self.quant_conv2(h)
        if self.encoder_normalize_embedding:
            h1 = F.normalize(h1, p=2, dim=1)
            h2 = F.normalize(h2, p=2, dim=1)

        ret_spatial = self.quantize(h1)
        ret_motion = self.quantize2(h2)

        ret_spatial['continuous'] = h1
        
        return ret_spatial, ret_motion
        
    def decode(self, quant):
        # distill_conv_out = self.post_quant_conv_distill(quant)
        quant2 = self.post_quant_conv(quant)
        rec = self.decoder(quant2)
        return rec

    def decode_motion(self, quant):
        quant2 = self.post_quant_conv2(quant)
        depth = self.decoder2(quant2)
        return depth
    
    def forward(self, input1, input2):
        gt_motion = self.gt_motion(input1, input2)
        encoded_spatial, encoded_motion = self.encode(input2)
        rec = self.decode(encoded_spatial["quantized"])
        motion = self.decode_motion(encoded_motion["quantized"])
        return rec, motion, (encoded_spatial['quantization_loss'], encoded_spatial['entropy_loss']), (encoded_motion['quantization_loss'], encoded_motion['entropy_loss']), gt_motion

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.decoder2.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quantize2.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.quant_conv2.parameters())+
                                  list(self.post_quant_conv.parameters())+
                                  list(self.post_quant_conv2.parameters()),
                                  lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))
        
        scheduler_ae_warmup = self.get_warmup_scheduler(opt_ae, self.loss.warmup_steps, self.min_lr_multiplier)        

        return [opt_ae], [scheduler_ae_warmup]
    
    def validation_step(self, batch, batch_idx):
        x1, x2 = self.get_input(batch)
        xrec, pred_motion, qloss1, qloss2, gt_motion = self(x1, x2)

        distill_loss = None
        aeloss, log_dict_ae = self.loss(qloss1, qloss2, distill_loss, x2, xrec, gt_motion, pred_motion, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        self.entropy_loss_weight_scheduling()
        self.entropy_loss_weight_scheduling_q2()
        self.log("train/enropy_loss_weight", self.loss.entropy_loss_weight, 
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        opt_ae = self.optimizers()
        scheduler_ae_warmup = self.lr_schedulers()
        
        x1, x2 = self.get_input(batch)
        
        xrec, pred_motion, qloss1, qloss2, gt_motion  = self(x1, x2)

        distill_loss = None

        optimizer_idx = 0
        aeloss, log_dict_ae = self.loss(qloss1, qloss2, distill_loss, x2, xrec, gt_motion, pred_motion, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        aeloss = aeloss / self.grad_acc_steps
        self.manual_backward(aeloss) 
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_ae.step()
            opt_ae.zero_grad()
            scheduler_ae_warmup.step()

    def log_images(self, batch, **kwargs):
        log = dict()
        x1, x2 = self.get_input(batch)
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        xrec, _, _, _, _ = self(x1, x2)
        log["input_t"] = x1
        log["input_t+1"] = x2
        log["reconstruction_t+1"] = xrec
        return log

class VQModel_2frame_motion(VQModel_1frame_motion):
    """
    VAE motion model
    """
    def __init__(
        self,
        encoder_config,
        decoder_config,
        decoder_motion_config,
        quantizer_config,
        motion_config,
        loss_config,
        grad_acc_steps=1,
        cont_ratio_trainig= 0.0,
        ignore_keys=None,
        monitor=None,
        entropy_loss_weight_scheduler_config=None,
        entropy_loss_weight_scheduler_config_q2=None,
        min_lr_multiplier=0.1,
        only_decoder=False,
        scale_equivariance=None,
    ):
        super().__init__(encoder_config, decoder_config, decoder_motion_config, quantizer_config, motion_config, loss_config,
                         grad_acc_steps, cont_ratio_trainig, ignore_keys, monitor, entropy_loss_weight_scheduler_config,
                         entropy_loss_weight_scheduler_config_q2, min_lr_multiplier, only_decoder, scale_equivariance,)       
    
    def encode(self, x1, x2):
        h = self.encoder(x1, x2)
        h1 = self.quant_conv(h)
        h2 = self.quant_conv2(h)
        if self.encoder_normalize_embedding:
            h1 = F.normalize(h1, p=2, dim=1)
            h2 = F.normalize(h2, p=2, dim=1)

        ret_spatial = self.quantize(h1)
        ret_motion = self.quantize2(h2)
        
        return ret_spatial, ret_motion
    
    def forward(self, input1, input2):
        gt_motion = self.gt_motion(input1, input2)
        encoded_spatial, encoded_motion = self.encode(input1, input2)
        rec = self.decode(encoded_spatial["quantized"])
        motion = self.decode_motion(encoded_motion["quantized"])
        return rec, motion, (encoded_spatial['quantization_loss'], encoded_spatial['entropy_loss']), (encoded_motion['quantization_loss'], encoded_motion['entropy_loss']), gt_motion
    
    def validation_step(self, batch, batch_idx):
        x1, x2 = self.get_input(batch)
        xrec, pred_motion, qloss1, qloss2, gt_motion = self(x1, x2)

        distill_loss = None
        aeloss, log_dict_ae = self.loss(qloss1, qloss2, distill_loss, x2, xrec, gt_motion, pred_motion, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        disp_loss = log_dict_ae["val/disp_loss"]
        self.log("val/disp_loss", disp_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)


class VQModel_1frame_depth(VQModel):
    """
    MAE depth model
    """
    def __init__(
        self,
        encoder_config,
        decoder_config,
        decoder_motion_config,
        quantizer_config,
        depth_config,
        loss_config,
        grad_acc_steps=1,
        cont_ratio_trainig= 0.0,
        ignore_keys=None,
        monitor=None,
        entropy_loss_weight_scheduler_config=None,
        entropy_loss_weight_scheduler_config_q2=None,
        min_lr_multiplier=0.1,
        only_decoder=False,
        scale_equivariance=None,
    ):
        super().__init__(encoder_config, decoder_config, quantizer_config, loss_config, grad_acc_steps,
                         cont_ratio_trainig, ignore_keys, monitor, entropy_loss_weight_scheduler_config,
                         min_lr_multiplier, only_decoder, scale_equivariance,)       

        self.quantize2 = instantiate_from_config(quantizer_config)  # motion quantizer
        self.decoder2 = instantiate_from_config(decoder_motion_config)     # motion decoder
        self.depth = instantiate_from_config(depth_config)
        self.entropy_loss_weight_scheduler_q2 = instantiate_from_config(entropy_loss_weight_scheduler_config_q2)
        
        
        # Convolutional layers for quantization
        self.quant_conv = nn.Conv2d(encoder_config.params["z_channels"], quantizer_config.params["e_dim"], 1)
        self.post_quant_conv = nn.Conv2d(quantizer_config.params["e_dim"], decoder_config.params["z_channels"], 1)

        self.quant_conv2 = nn.Conv2d(encoder_config.params["z_channels"], quantizer_config.params["e_dim"], 1)
        self.post_quant_conv2 = nn.Conv2d(quantizer_config.params["e_dim"], decoder_motion_config.params["z_channels"], 1)

    def entropy_loss_weight_scheduling_q2(self):
        self.loss.entropy_loss_weight_q2 = self.entropy_loss_weight_scheduler_q2(self.global_step)
    
    def gt_depth(self, x):
        h = self.depth(x)         # [B, N, 1]
        return h
    
    def encode(self, x):
        h = self.encoder(x)
        h1 = self.quant_conv(h)
        h2 = self.quant_conv2(h)
        if self.encoder_normalize_embedding:
            h1 = F.normalize(h1, p=2, dim=1)
            h2 = F.normalize(h2, p=2, dim=1)

        ret_spatial = self.quantize(h1)
        ret_depth = self.quantize2(h2)

        ret_spatial["continuous"] = h1
        
        return ret_spatial, ret_depth
        
    def decode(self, quant):
        # distill_conv_out = self.post_quant_conv_distill(quant)
        quant2 = self.post_quant_conv(quant)
        rec = self.decoder(quant2)
        return rec

    def decode_depth(self, quant):
        quant2 = self.post_quant_conv2(quant)
        depth = self.decoder2(quant2)
        return depth
    
    def forward(self, input):
        gt_depth = self.gt_depth(input)
        encoded_spatial, encoded_depth = self.encode(input)
        rec = self.decode(encoded_spatial["quantized"])
        depth = self.decode_depth(encoded_depth["quantized"])
        return rec, depth, (encoded_spatial['quantization_loss'], encoded_spatial['entropy_loss']), (encoded_depth['quantization_loss'], encoded_depth['entropy_loss']), gt_depth

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.decoder2.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quantize2.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.quant_conv2.parameters())+
                                  list(self.post_quant_conv.parameters())+
                                  list(self.post_quant_conv2.parameters()),
                                  lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))
        
        scheduler_ae_warmup = self.get_warmup_scheduler(opt_ae, self.loss.warmup_steps, self.min_lr_multiplier)        

        return [opt_ae], [scheduler_ae_warmup]
    
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch)
        xrec, pred_depth, qloss1, qloss2, gt_depth = self(x)

        distill_loss = None
        aeloss, log_dict_ae = self.loss(qloss1, qloss2, distill_loss, x, xrec, gt_depth, pred_depth, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        self.entropy_loss_weight_scheduling()
        self.entropy_loss_weight_scheduling_q2()
        self.log("train/enropy_loss_weight", self.loss.entropy_loss_weight, 
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        opt_ae = self.optimizers()
        scheduler_ae_warmup = self.lr_schedulers()
        
        x = self.get_input(batch)
        
        xrec, pred_depth, qloss1, qloss2, gt_depth  = self(x)

        distill_loss = None

        optimizer_idx = 0
        aeloss, log_dict_ae = self.loss(qloss1, qloss2, distill_loss, x, xrec, gt_depth, pred_depth, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        aeloss = aeloss / self.grad_acc_steps
        self.manual_backward(aeloss) 
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_ae.step()
            opt_ae.zero_grad()
            scheduler_ae_warmup.step()

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch)
        x = x.to(self.device)
        xrec, _, _, _, _ = self(x)
        log["input"] = x
        log["reconstruction_t+1"] = xrec
        return log

class VQModel_1frame_dino(VQModel):
    """
    Dino model distillation
    """
    def __init__(
        self,
        encoder_config,
        decoder_config,
        decoder_dino_config,
        quantizer_config,
        dino_config,
        loss_config,
        grad_acc_steps=1,
        cont_ratio_trainig= 0.0,
        ignore_keys=None,
        monitor=None,
        entropy_loss_weight_scheduler_config=None,
        entropy_loss_weight_scheduler_config_q2=None,
        min_lr_multiplier=0.1,
        only_decoder=False,
        scale_equivariance=None,
    ):
        super().__init__(encoder_config, decoder_config, quantizer_config, loss_config, grad_acc_steps,
                         cont_ratio_trainig, ignore_keys, monitor, entropy_loss_weight_scheduler_config,
                         min_lr_multiplier, only_decoder, scale_equivariance,)       

        self.quantize2 = instantiate_from_config(quantizer_config)  # motion quantizer
        self.decoder2 = instantiate_from_config(decoder_dino_config)     # motion decoder
        self.dino = instantiate_from_config(dino_config)
        self.entropy_loss_weight_scheduler_q2 = instantiate_from_config(entropy_loss_weight_scheduler_config_q2)
        
        # Convolutional layers for quantization
        self.quant_conv = nn.Conv2d(encoder_config.params["z_channels"], quantizer_config.params["e_dim"], 1)
        self.post_quant_conv = nn.Conv2d(quantizer_config.params["e_dim"], decoder_config.params["z_channels"], 1)

        self.quant_conv2 = nn.Conv2d(encoder_config.params["z_channels"], quantizer_config.params["e_dim"], 1)
        self.post_quant_conv2 = nn.Conv2d(quantizer_config.params["e_dim"], decoder_dino_config.params["z_channels"], 1)

    def entropy_loss_weight_scheduling_q2(self):
        self.loss.entropy_loss_weight_q2 = self.entropy_loss_weight_scheduler_q2(self.global_step)
    
    def gt_dino(self, x):
        h = self.dino(x)         # [B, N, 1]
        return h
    
    def encode(self, x):
        h = self.encoder(x)
        h1 = self.quant_conv(h)
        h2 = self.quant_conv2(h)
        if self.encoder_normalize_embedding:
            h1 = F.normalize(h1, p=2, dim=1)
            h2 = F.normalize(h2, p=2, dim=1)

        ret_spatial = self.quantize(h1)
        ret_depth = self.quantize2(h2)

        ret_spatial["continuous"] = h1
        
        return ret_spatial, ret_depth
        
    def decode(self, quant):
        # distill_conv_out = self.post_quant_conv_distill(quant)
        quant2 = self.post_quant_conv(quant)
        rec = self.decoder(quant2)
        return rec

    def decode_dino(self, quant):
        quant2 = self.post_quant_conv2(quant)
        dino = self.decoder2(quant2)
        return dino
    
    def forward(self, input):
        gt_dino = self.gt_dino(input)
        encoded_spatial, encoded_dino = self.encode(input)
        rec = self.decode(encoded_spatial["quantized"])
        dino = self.decode_dino(encoded_dino["quantized"])
        return rec, dino, (encoded_spatial['quantization_loss'], encoded_spatial['entropy_loss']), (encoded_dino['quantization_loss'], encoded_dino['entropy_loss']), gt_dino

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.decoder2.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quantize2.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.quant_conv2.parameters())+
                                  list(self.post_quant_conv.parameters())+
                                  list(self.post_quant_conv2.parameters()),
                                  lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))
        
        scheduler_ae_warmup = self.get_warmup_scheduler(opt_ae, self.loss.warmup_steps, self.min_lr_multiplier)        

        return [opt_ae], [scheduler_ae_warmup]
    
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch)
        xrec, pred_dino, qloss1, qloss2, gt_dino = self(x)

        distill_loss = None
        aeloss, log_dict_ae = self.loss(qloss1, qloss2, distill_loss, x, xrec, gt_dino, pred_dino, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        dino_loss = log_dict_ae["val/dino_loss"]
        self.log("val/dino_loss", dino_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        

    def training_step(self, batch, batch_idx):
        self.entropy_loss_weight_scheduling()
        self.entropy_loss_weight_scheduling_q2()
        self.log("train/enropy_loss_weight", self.loss.entropy_loss_weight, 
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        opt_ae = self.optimizers()
        scheduler_ae_warmup = self.lr_schedulers()
        
        x = self.get_input(batch)
        
        xrec, pred_dino, qloss1, qloss2, gt_dino  = self(x)

        distill_loss = None

        optimizer_idx = 0
        aeloss, log_dict_ae = self.loss(qloss1, qloss2, distill_loss, x, xrec, gt_dino, pred_dino, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        aeloss = aeloss / self.grad_acc_steps
        self.manual_backward(aeloss) 
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_ae.step()
            opt_ae.zero_grad()
            scheduler_ae_warmup.step()

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch)
        x = x.to(self.device)
        xrec, _, _, _, _ = self(x)
        log["input"] = x
        log["reconstruction_t+1"] = xrec
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
        ret["continuous"] = h
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
    
class VQModel_multiframes_2dgrid(VQModel_multiframes):
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
        # print("Encoder output: ", h.shape)
        h = self.quant_conv(h)
        if self.encoder_normalize_embedding:
            h = F.normalize(h, p=2, dim=1)
        ret = self.quantize(h)
        ret["continuous"] = h
        return ret
        
    def decode(self, quant):
        # distill_conv_out = self.post_quant_conv_distill(quant)
        quant2 = self.post_quant_conv(quant)
        rec1, rec2 = self.decoder(quant2)
        return rec1, rec2
    
    def forward(self, input1, input2):
        encoded = self.encode(input1, input2)
        rec1, rec2 = self.decode(encoded["quantized"])
        return rec1, rec2, (encoded['quantization_loss'], encoded['entropy_loss'])
    
    def validation_step(self, batch, batch_idx):
        x1, x2 = self.get_input(batch)
        xrec1, xrec2, qloss = self(x1, x2)

        distill_loss = None
        optimizer_idx = 0
        aeloss, log_dict_ae = self.loss(qloss, distill_loss, x1, x2, xrec1, xrec2, optimizer_idx, self.global_step,
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
        
        xrec1, xrec2, qloss = self(x1, x2)

        distill_loss = None

        optimizer_idx = 0
        aeloss, log_dict_ae = self.loss(qloss, distill_loss, x1, x2, xrec1, xrec2, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        aeloss = aeloss / self.grad_acc_steps
        self.manual_backward(aeloss) 
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_ae.step()
            opt_ae.zero_grad()
            scheduler_ae_warmup.step()
        
    def log_images(self, batch, **kwargs):
        log = dict()
        x1, x2 = self.get_input(batch)
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        xrec1, xrec2, _ = self(x1, x2)
        log["input_t"] = x1
        log["input_t+1"] = x2
        log["reconstruction_t"] = xrec1
        log["reconstruction_t+1"] = xrec2
        return log

class VQModel_multiframes_motionModel(VQModel_multiframes):
    """
    MAE motion model
    """
    def __init__(
        self,
        encoder_config,
        decoder_config,
        decoder_motion_config,
        quantizer_config,
        raft_config,
        loss_config,
        grad_acc_steps=1,
        cont_ratio_trainig= 0.0,
        ignore_keys=None,
        monitor=None,
        entropy_loss_weight_scheduler_config=None,
        entropy_loss_weight_scheduler_config_q2=None,
        min_lr_multiplier=0.1,
        only_decoder=False,
        scale_equivariance=None,
    ):
        super().__init__(encoder_config, decoder_config, quantizer_config, loss_config, grad_acc_steps,
                         cont_ratio_trainig, ignore_keys, monitor, entropy_loss_weight_scheduler_config,
                         min_lr_multiplier, only_decoder, scale_equivariance,)       

        self.quantize2 = instantiate_from_config(quantizer_config)  # motion quantizer
        self.decoder2 = instantiate_from_config(decoder_motion_config)     # motion decoder
        self.raft_displacement = instantiate_from_config(raft_config)
        self.entropy_loss_weight_scheduler = instantiate_from_config(entropy_loss_weight_scheduler_config)
        self.entropy_loss_weight_scheduler_q2 = instantiate_from_config(entropy_loss_weight_scheduler_config_q2)
        
        # Convolutional layers for quantization
        self.quant_conv = nn.Conv2d(encoder_config.params["z_channels"], quantizer_config.params["e_dim"], 1)
        self.post_quant_conv = nn.Conv2d(quantizer_config.params["e_dim"], decoder_config.params["z_channels"], 1)

        self.quant_conv2 = nn.Conv2d(encoder_config.params["z_channels"], quantizer_config.params["e_dim"], 1)
        self.post_quant_conv2 = nn.Conv2d(quantizer_config.params["e_dim"], decoder_motion_config.params["z_channels"], 1)

    def entropy_loss_weight_scheduling(self):
        self.loss.entropy_loss_weight = self.entropy_loss_weight_scheduler(self.global_step)

    def entropy_loss_weight_scheduling_q2(self):
        self.loss.entropy_loss_weight_q2 = self.entropy_loss_weight_scheduler_q2(self.global_step)

    def gt_displacement(self, x1, x2):
        h = self.raft_displacement(x1, x2)         # [B, N, 2]
        
        return h
    
    def encode(self, x1, x2):
        h = self.encoder(x1, x2)
        h1 = self.quant_conv(h)
        h2 = self.quant_conv2(h)
        if self.encoder_normalize_embedding:
            h1 = F.normalize(h1, p=2, dim=1)
            h2 = F.normalize(h2, p=2, dim=1)

        ret_spatial = self.quantize(h1)
        ret_motion = self.quantize2(h2)

        ret_spatial["continuous"] = h1
        
        return ret_spatial, ret_motion
        
    def decode(self, quant):
        # distill_conv_out = self.post_quant_conv_distill(quant)
        quant2 = self.post_quant_conv(quant)
        rec = self.decoder(quant2)
        return rec

    def decode_motion(self, quant):
        quant2 = self.post_quant_conv2(quant)
        disp = self.decoder2(quant2)
        return disp
    
    def forward(self, input1, input2):
        gt_disp = self.gt_displacement(input1, input2)
        encoded_spatial, encoded_motion = self.encode(input1, input2)
        rec = self.decode(encoded_spatial["quantized"])
        motion = self.decode_motion(encoded_motion["quantized"])
        return rec, motion, (encoded_spatial['quantization_loss'], encoded_spatial['entropy_loss']), (encoded_motion['quantization_loss'], encoded_motion['entropy_loss']), gt_disp

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.decoder2.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quantize2.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.quant_conv2.parameters())+
                                  list(self.post_quant_conv.parameters())+
                                  list(self.post_quant_conv2.parameters()),
                                  lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))
        
        scheduler_ae_warmup = self.get_warmup_scheduler(opt_ae, self.loss.warmup_steps, self.min_lr_multiplier)        

        return [opt_ae], [scheduler_ae_warmup]
    
    def validation_step(self, batch, batch_idx):
        x1, x2 = self.get_input(batch)
        xrec, pred_disp, qloss1, qloss2, gt_disp = self(x1, x2)

        distill_loss = None
        aeloss, log_dict_ae = self.loss(qloss1, qloss2, distill_loss, x2, xrec, gt_disp, pred_disp, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        self.entropy_loss_weight_scheduling()
        self.entropy_loss_weight_scheduling_q2()
        self.log("train/enropy_loss_weight", self.loss.entropy_loss_weight, 
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        opt_ae = self.optimizers()
        scheduler_ae_warmup = self.lr_schedulers()
        
        x1, x2 = self.get_input(batch)
        
        xrec, pred_disp, qloss1, qloss2, gt_disp  = self(x1, x2)

        distill_loss = None

        optimizer_idx = 0
        aeloss, log_dict_ae = self.loss(qloss1, qloss2, distill_loss, x2, xrec, gt_disp, pred_disp, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        aeloss = aeloss / self.grad_acc_steps
        self.manual_backward(aeloss) 
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_ae.step()
            opt_ae.zero_grad()
            scheduler_ae_warmup.step()

    def log_images(self, batch, **kwargs):
        log = dict()
        x1, x2 = self.get_input(batch)
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        xrec, _, _, _, _ = self(x1, x2)
        log["input_t"] = x1
        log["input_t+1"] = x2
        log["reconstruction_t+1"] = xrec
        return log

class VQModel_multiframes_depthModel(VQModel_multiframes):
    """
    MAE motion model
    """
    def __init__(
        self,
        encoder_config,
        decoder_config,
        decoder_motion_config,
        quantizer_config,
        depth_config,
        loss_config,
        grad_acc_steps=1,
        cont_ratio_trainig= 0.0,
        ignore_keys=None,
        monitor=None,
        entropy_loss_weight_scheduler_config=None,
        entropy_loss_weight_scheduler_config_q2=None,
        min_lr_multiplier=0.1,
        only_decoder=False,
        scale_equivariance=None,
    ):
        super().__init__(encoder_config, decoder_config, quantizer_config, loss_config, grad_acc_steps,
                         cont_ratio_trainig, ignore_keys, monitor, entropy_loss_weight_scheduler_config,
                         min_lr_multiplier, only_decoder, scale_equivariance,)       

        self.quantize2 = instantiate_from_config(quantizer_config)  # motion quantizer
        self.decoder2 = instantiate_from_config(decoder_motion_config)     # motion decoder
        self.depth = instantiate_from_config(depth_config)
        self.entropy_loss_weight_scheduler_q2 = instantiate_from_config(entropy_loss_weight_scheduler_config_q2)
        
        # Convolutional layers for quantization
        self.quant_conv = nn.Conv2d(encoder_config.params["z_channels"], quantizer_config.params["e_dim"], 1)
        self.post_quant_conv = nn.Conv2d(quantizer_config.params["e_dim"], decoder_config.params["z_channels"], 1)

        self.quant_conv2 = nn.Conv2d(encoder_config.params["z_channels"], quantizer_config.params["e_dim"], 1)
        self.post_quant_conv2 = nn.Conv2d(quantizer_config.params["e_dim"], decoder_motion_config.params["z_channels"], 1)

    def entropy_loss_weight_scheduling(self):
        self.loss.entropy_loss_weight = self.entropy_loss_weight_scheduler(self.global_step)

    def entropy_loss_weight_scheduling_q2(self):
        self.loss.entropy_loss_weight_q2 = self.entropy_loss_weight_scheduler_q2(self.global_step)

    def gt_depth(self, x):
        h = self.depth(x)         # [B, N, 1]
        return h
    
    def encode(self, x1, x2):
        h = self.encoder(x1, x2)
        h1 = self.quant_conv(h)
        h2 = self.quant_conv2(h)
        if self.encoder_normalize_embedding:
            h1 = F.normalize(h1, p=2, dim=1)
            h2 = F.normalize(h2, p=2, dim=1)

        ret_spatial = self.quantize(h1)
        ret_depth = self.quantize2(h2)
        
        return ret_spatial, ret_depth
        
    def decode(self, quant):
        # distill_conv_out = self.post_quant_conv_distill(quant)
        quant2 = self.post_quant_conv(quant)
        rec = self.decoder(quant2)
        return rec

    def decode_depth(self, quant):
        quant2 = self.post_quant_conv2(quant)
        disp = self.decoder2(quant2)
        return disp
    
    def forward(self, input1, input2):
        gt_depth = self.gt_depth(input2)
        # print("GT shape : ", gt_depth.shape)
        encoded_spatial, encoded_depth = self.encode(input1, input2)
        rec = self.decode(encoded_spatial["quantized"])
        # print("Rec shape : ", rec.shape)
        depth = self.decode_depth(encoded_depth["quantized"])
        # print("Pred depth : ", depth.shape)
        return rec, depth, (encoded_spatial['quantization_loss'], encoded_spatial['entropy_loss']), (encoded_depth['quantization_loss'], encoded_depth['entropy_loss']), gt_depth

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.decoder2.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quantize2.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.quant_conv2.parameters())+
                                  list(self.post_quant_conv.parameters())+
                                  list(self.post_quant_conv2.parameters()),
                                  lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))
        
        scheduler_ae_warmup = self.get_warmup_scheduler(opt_ae, self.loss.warmup_steps, self.min_lr_multiplier)        

        return [opt_ae], [scheduler_ae_warmup]
    
    def validation_step(self, batch, batch_idx):
        x1, x2 = self.get_input(batch)
        xrec, pred_depth, qloss1, qloss2, gt_depth = self(x1, x2)

        distill_loss = None
        aeloss, log_dict_ae = self.loss(qloss1, qloss2, distill_loss, x2, xrec, gt_depth, pred_depth, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        depth_loss = log_dict_ae["val/depth_loss"]
        self.log("val/depth_loss", depth_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        self.entropy_loss_weight_scheduling()
        self.entropy_loss_weight_scheduling_q2()
        self.log("train/enropy_loss_weight", self.loss.entropy_loss_weight, 
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        opt_ae = self.optimizers()
        scheduler_ae_warmup = self.lr_schedulers()
        
        x1, x2 = self.get_input(batch)
        
        xrec, pred_depth, qloss1, qloss2, gt_depth  = self(x1, x2)

        distill_loss = None

        optimizer_idx = 0
        aeloss, log_dict_ae = self.loss(qloss1, qloss2, distill_loss, x2, xrec, gt_depth, pred_depth, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        aeloss = aeloss / self.grad_acc_steps
        self.manual_backward(aeloss) 
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_ae.step()
            opt_ae.zero_grad()
            scheduler_ae_warmup.step()

    def log_images(self, batch, **kwargs):
        log = dict()
        x1, x2 = self.get_input(batch)
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        xrec, _, _, _, _ = self(x1, x2)
        log["input_t"] = x1
        log["input_t+1"] = x2
        log["reconstruction_t+1"] = xrec
        return log

class VQModel_multiframes_dino(VQModel_multiframes_motionModel):
    """
    MAE motion model
    """
    def __init__(
        self,
        encoder_config,
        decoder_config,
        decoder_dino_config,
        quantizer_config,
        dino_config,
        loss_config,
        grad_acc_steps=1,
        cont_ratio_trainig= 0.0,
        ignore_keys=None,
        monitor=None,
        entropy_loss_weight_scheduler_config=None,
        entropy_loss_weight_scheduler_config_q2=None,
        min_lr_multiplier=0.1,
        only_decoder=False,
        scale_equivariance=None,
        **kwargs,
    ):
        super().__init__(encoder_config, decoder_config, decoder_motion_config=decoder_dino_config, quantizer_config=quantizer_config, 
                         raft_config=dino_config, loss_config=loss_config, grad_acc_steps=grad_acc_steps, cont_ratio_trainig=cont_ratio_trainig,
                         ignore_keys=ignore_keys, monitor=monitor, entropy_loss_weight_scheduler_config=entropy_loss_weight_scheduler_config,
                         entropy_loss_weight_scheduler_config_q2=entropy_loss_weight_scheduler_config_q2, min_lr_multiplier=min_lr_multiplier, 
                         only_decoder=only_decoder, scale_equivariance=scale_equivariance, **kwargs)       

        self.dino = instantiate_from_config(dino_config)
        self.decoder2 = instantiate_from_config(decoder_dino_config)

    def gt_dino(self, x2):
        h = self.dino(x2)         # [B, N, 2]
        return h

    def decode(self, quant):
        # distill_conv_out = self.post_quant_conv_distill(quant)
        quant2 = self.post_quant_conv(quant)
        rec1, rec2 = self.decoder(quant2)
        return rec1, rec2

    def forward(self, input1, input2):
        gt_dino = self.gt_dino(input2)
        encoded_spatial, encoded_dino = self.encode(input1, input2)
        rec1, rec2 = self.decode(encoded_spatial["quantized"])
        dino = self.decode_motion(encoded_dino["quantized"])
        return rec1, rec2, dino, (encoded_spatial['quantization_loss'], encoded_spatial['entropy_loss']), (encoded_dino['quantization_loss'], encoded_dino['entropy_loss']), gt_dino

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.decoder2.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quantize2.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.quant_conv2.parameters())+
                                  list(self.post_quant_conv.parameters())+
                                  list(self.post_quant_conv2.parameters()),
                                  lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))
        
        scheduler_ae_warmup = self.get_warmup_scheduler(opt_ae, self.loss.warmup_steps, self.min_lr_multiplier)        

        return [opt_ae], [scheduler_ae_warmup]
    
    def validation_step(self, batch, batch_idx):
        x1, x2 = self.get_input(batch)
        xrec1, xrec2, pred_dino, qloss1, qloss2, gt_dino = self(x1, x2)

        distill_loss = None
        aeloss, log_dict_ae = self.loss(qloss1, qloss2, distill_loss, x1, x2, xrec1, xrec2, gt_dino, pred_dino, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        dino_loss = log_dict_ae["val/dino_loss"]
        self.log("val/dino_loss", dino_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        self.entropy_loss_weight_scheduling()
        self.entropy_loss_weight_scheduling_q2()
        self.log("train/enropy_loss_weight", self.loss.entropy_loss_weight, 
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        opt_ae = self.optimizers()
        scheduler_ae_warmup = self.lr_schedulers()
        
        x1, x2 = self.get_input(batch)
        
        xrec1, xrec2, pred_dino, qloss1, qloss2, gt_dino = self(x1, x2)

        distill_loss = None

        optimizer_idx = 0
        aeloss, log_dict_ae = self.loss(qloss1, qloss2, distill_loss, x1, x2, xrec1, xrec2, gt_dino, pred_dino, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        aeloss = aeloss / self.grad_acc_steps
        self.manual_backward(aeloss) 
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_ae.step()
            opt_ae.zero_grad()
            scheduler_ae_warmup.step()

    def log_images(self, batch, **kwargs):
        log = dict()
        x1, x2 = self.get_input(batch)
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        xrec1, xrec2, _, _, _, _ = self(x1, x2)
        log["input_t"] = x1
        log["input_t+1"] = x2
        log["reconstruction_t"] = xrec1
        log["reconstruction_t+1"] = xrec2
        return log


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