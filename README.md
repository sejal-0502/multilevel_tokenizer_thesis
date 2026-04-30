# Multilevel Representation Learning for DiT-based Image Generation

This repository hosts the code for tokenizers designed to preserve both semantic and perceptual information in continuous latent representations. It includes implementations of Masked Autoencoders (MAE), Temporal MAE, Temporal Compression, Auxiliary supervision using pretrained models (DINO, Depth, RAFT, MIM), and EMA-stabilized teacher supervision.

<table>
  <tr>
    <td align="center"><img src="imgs/model_perf.png" width="600"/></td>
    <td align="center"><img src="imgs/visuals.png" width="600"/></td>
  </tr>
  <tr>
    <td align="center">Figure 1 : Performance of tokenizer across all 4 evaluation axes</td>
    <td align="center">Figuare 2 : Example Visuals for each tokenizer model</td>
  </tr>
</table>

## Contributions 

We explore multiple self-supervised learning methods for designing a tokenizer that improves semantic quality without significantly degrading perceptual fidelity compared to a standard VQ-VAE baseline. The proposed tokenizers improve semantic and geometric representations, with auxiliary supervision methods showing the strongest generative performance.

Overall, our experiments show that self-supervised approaches, including masking, temporal context modeling, knowledge distillation from strong pretrained priors, and EMA-stabilized supervision, can effectively balance semantic and perceptual objectives in tokenizer learning.

## Checkpoints & Results

Results for Multilevel Representation Learning. Checkpoints are located in the `checkpoints/` folder.

| Model | rFID ↓ | gFID ↓ | mIoU ↑ | RMSE ↓ |
| :--- | :---: | :---: | :---: | :---: |
| VQ-VAE (Baseline) | 29.09 | 45.09 | 34.32 | 6.4156 | 
| MAE | 32.43 | 50.88 | 36.47 | 6.0660 |
| Temporal MAE | 31.86 | 55.10 | 35.46 | 6.1020 |  
| **Dino Distill.** | **29.60** | **44.03** | **44.34** | **5.9185** |
| Depth Distill. | 30.70 | 44.60 | 38.39 | 6.1160 |
| Raft Distill. | 30.87 | 45.31 | 36.10 | 6.0661 |
| EMA Supervision | 33.52 | 49.29 | 35.25 | 6.1494 |

> *Note: We report scores for our best configurations. Detailed discussion on mask ratios and architectural designs can be found in the attached report.*

## Requirements
A suitable python environment can be created and activated with:

```
conda env create -f environment.yaml
conda activate venv
```

## Model Training

### Training a tokenizer
```
python main.py --base configs/tokenizer.yaml -t True --n_gpus=4 --enable_codebook_usage_logger 
```
We select the effective batch size of 128 for all our experiments.

### Training a generation model
```
python main.py --base configs/generation.yaml -t True --n_gpus=4
```

The generation code is used solely for evaluation purposes and remains unchanged. 
Original repository reference : [Orbis](https://github.com/lmb-freiburg/orbis.git)

## Model Evaluations

> *Follow the configs from the checkpoints attached.*

**Perceptual Quality (rFID) :**
```
python evaluate/compute_codebook_usage.py --config_path /work/dlclarge2/mutakeks-titok/Thesis/mae_orbis/baseline/config.yaml --ckpt_path /work/dlclarge2/mutakeks-titok/Thesis/mae_orbis/baseline/checkpoints/last.ckpt --compute_rFID_score
```

**Generative Quality (gFID) :**
```
python evaluate/compute_codebook_usage_dit.py --config_path /work/dlclarge2/mutakeks-titok/Thesis/mae_orbis/baseline/config.yaml --ckpt_path /work/dlclarge2/mutakeks-titok/Thesis/mae_orbis/baseline/checkpoints/last.ckpt --compute_rFID_score
```

**Semantic Segmentation Probe (mIoU) :**
```
# python evaluate/segmentation.py --ckpt /work/dlclarge2/mutakeks-titok/Thesis/mae_orbis/baseline/checkpoints/last.ckpt --config /work/dlclarge2/mutakeks-titok/Thesis/mae_orbis/baseline/config.yaml --dump_vis True 
```

**Depth Estimation Probe (RMSE) :**
```
# python evaluate/depth.py --ckpt /work/dlclarge2/mutakeks-titok/Thesis/mae_orbis/baseline/checkpoints/last.ckpt --config /work/dlclarge2/mutakeks-titok/Thesis/mae_orbis/baseline/config.yaml --split_dir evaluate/data --dump_vis True 
```
> *Note: For 2 frame setups (like Temporal MAE), '--num_input_frames' needs to be set manually to 2 for probe scripts. The code defauls to 1 input frame only.*