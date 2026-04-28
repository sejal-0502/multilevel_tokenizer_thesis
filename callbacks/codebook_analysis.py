from typing import Any
import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback
import numpy as np
from einops import rearrange
from pytorch_lightning.utilities import rank_zero_only


class CodebookUsageLogger(Callback):
    """
    Track unique codebook indices used by a vector-quantizer during training/validation.
    """
    def __init__(self, log_batches_training: bool = False, quantizer_attr: str = "quantize"):
        super().__init__()
        self.log_batches_training = log_batches_training
        self.quantizer_attr = quantizer_attr
        self.used_indices_train = set()
        self.used_indices_val = set()
        self.hook_dict = {}
        self.hook_handle = None

    def setup(self, trainer: Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        # setup fwd hook to collect indices
        quantizer = getattr(pl_module, self.quantizer_attr)
        def quantizer_hook_fn(module, x, y):
            # _, _, info = y
            quant = y
            info = quant["indices"]
            if isinstance(info[2], tuple):
                self.hook_dict["indices"] = info[2][0].cpu().numpy()
                self.hook_dict["indices2"] = info[2][1].cpu().numpy()
            else:
                self.hook_dict["indices"] = info[2].cpu().numpy()
        
        if isinstance(quantizer, torch.nn.ModuleList):
            self.hook_handle = quantizer[0].register_forward_hook(quantizer_hook_fn)
        else:
            self.hook_handle = quantizer.register_forward_hook(quantizer_hook_fn)

    # update logger's counter after iter, log for batch if requested
    def on_train_batch_end(self, trainer: Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        used_batch_set = set(self.hook_dict['indices'].flatten())
        self.used_indices_train.update(used_batch_set)
        if self.log_batches_training:
            pl_module.log(f"train/indices_used_batch_avg_{self.quantizer_attr}", len(used_batch_set), prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=True)

    def on_validation_batch_end(self, trainer: Trainer, pl_module: pl.LightningModule, outputs:Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        used_batch_set = set(self.hook_dict['indices'].flatten())
        self.used_indices_val.update(used_batch_set)

    # log and reset counter
    def on_train_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.log(f"train/total_indices_used_{self.quantizer_attr}", len(self.used_indices_train), prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.used_indices_train = set()
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.log(f"val/total_indices_used_{self.quantizer_attr}", len(self.used_indices_val), prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.used_indices_val = set()


class CodebookTSNELogger(Callback):
    def __init__(self, epoch_frequency, num_embeddings_to_plot=4096, quantizer_attr: str = "quantize"):
        self.epoch_frequency = epoch_frequency
        self.quantizer_attr = quantizer_attr
        self.num_embeddings_to_plot = num_embeddings_to_plot
        self.embeddings_list = []
        self.hook_handle = None
        self.hook_dict = {}
    
    def setup(self, trainer: Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        # setup fwd hook to collect embeddings
        quantizer = getattr(pl_module, self.quantizer_attr)
        def embeddings_hook_fn(module, x, y):
            self.hook_dict["embeddings"] = x[0].detach().cpu()
        
        if isinstance(quantizer, torch.nn.ModuleList):
            self.hook_handle = quantizer[0].register_forward_hook(embeddings_hook_fn)
        else:
            self.hook_handle = quantizer.register_forward_hook(embeddings_hook_fn)
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.current_epoch % self.epoch_frequency == 0:
            # append embeddings if the total number of embeddings is less than the number of embeddings to plot
            total_embeddings = sum([len(embeddings) for embeddings in self.embeddings_list])
            if total_embeddings < self.num_embeddings_to_plot:
                self.embeddings_list.append(rearrange(self.hook_dict['embeddings'], "b c h w -> (b h w) c"))
    
    def get_tsned_projections(self, pl_module):
        from sklearn.manifold import TSNE
        quantizer = getattr(pl_module, self.quantizer_attr)
        tsne = TSNE(n_components=2, verbose=1, perplexity=40)
        embeddings = torch.cat(self.embeddings_list, dim=0)
        print("Shape of embeddings : ", embeddings.shape)
        if "Multires" in pl_module.__class__.__name__:
            codewords = quantizer[0].embedding.weight.cpu()
        else:
            codewords = quantizer.embedding.weight.cpu()
        print("Shape of codeworks : ", codewords.shape)
        all_vectors = torch.cat([embeddings, codewords], dim=0)
        tsne_results = tsne.fit_transform(all_vectors)
        tsne_embeddings, tsne_codewords = tsne_results[:len(embeddings)], tsne_results[len(embeddings):]
        return tsne_embeddings, tsne_codewords

    @staticmethod
    def get_plot_as_rgb(tsne_projection_embedding, tsne_projection_codewords):
        import matplotlib.pyplot as plt
        # Create a plot
        plt.scatter(tsne_projection_embedding[:, 0], tsne_projection_embedding[:, 1], c='b', label='Embeddings', s=0.5)
        plt.scatter(tsne_projection_codewords[:, 0], tsne_projection_codewords[:, 1], c='r', label='Codewords', s=0.5)
        plt.legend()

        # Render the plot onto a Matplotlib figure canvas
        fig = plt.gcf()
        fig.canvas.draw()

        # Get the RGB array from the figure canvas
        buf = fig.canvas.buffer_rgba()
        rgb_array = np.asarray(buf)[:, :, :3]

        plt.close()
        return rgb_array

    @rank_zero_only
    def log_tensorboard(self, pl_module):
        tsne_embeddings, tsne_codewords = self.get_tsned_projections(pl_module)
        plot = self.get_plot_as_rgb(tsne_embeddings, tsne_codewords)
        pl_module.logger.experiment.add_image(f"TSNE_{self.quantizer_attr}", plot, global_step=pl_module.global_step, dataformats="HWC")

    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch % self.epoch_frequency == 0:
            self.log_tensorboard(pl_module)
        self.embeddings_list = []