# stdlib
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

# third-party
import albumentations as A
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from util import instantiate_from_config  # noqa: E402

IGNORE_INDEX: int = -100


def mkdir(path: os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def resolve_cache_dtype(name: str) -> torch.dtype:
    CACHE_DTYPE_MAP: Dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    key = name.lower()
    if key not in CACHE_DTYPE_MAP:
        valid = ", ".join(sorted(CACHE_DTYPE_MAP))
        raise ValueError(f"Unsupported cache dtype '{name}'. Valid options: {valid}")
    return CACHE_DTYPE_MAP[key]


class KITTIDepthEigen(Dataset):
    def __init__(self, root_dir: str, split_file: str, transform=None, size: Tuple[int, int] | None = None):
        """
        root_dir: path to the KITTI dataset folder
        split_file: path to the eigen split file (e.g., eigen_train_files.txt)
        transform: albumentations transform
        size: target image size (height, width)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform if transform is not None else self.default_transform(size)

        # Read the split file
        with open(split_file, "r") as f:
            self.samples = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        # Split line into sequence path, frame index, camera side
        line = self.samples[idx]
        seq_path, frame_idx, cam_side = line.split()
        frame_idx = int(frame_idx)

        # Parse "<date>/<drive_sync>"
        date_folder, drive_folder = seq_path.split("/", 1)  # e.g. "2011_09_26", "2011_09_26_drive_0001_sync"

        # Determine camera folder based on side
        cam_folder = "image_02" if cam_side == "l" else "image_03"

        # ---- RGB path (your raw_data layout) ----
        img_path = (self.root_dir / "raw_data" / "image" / date_folder / drive_folder / cam_folder / "data" / f"{frame_idx:010d}.png")
        # ---- Depth path (your annotated depth layout) ----
        depth_path = (self.root_dir / "data_depth_annotated" / "all" / drive_folder / "proj_depth" / "groundtruth" / cam_folder / f"{frame_idx:010d}.png")

        # Load image and depth
        img = Image.open(img_path).convert("RGB")
        depth_pil = Image.open(depth_path)
        depth_np = np.array(depth_pil).astype(np.float32) / 256.0
        depth_np[depth_np == 0] = np.nan  # Mask missing depth

        # Apply transform
        transformed = self.transform(image=np.array(img), mask=depth_np)
        img_t = transformed["image"]
        depth_t = transformed["mask"]

        return img_t, torch.as_tensor(depth_t, dtype=torch.float32)

    @staticmethod
    def default_transform(size: Tuple[int, int] | None = None):
        if size is None:
            size = (352, 1216)
        return A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={"mask": "mask"})


# Cached feature dataset
class CachedFeatureDataset(Dataset):
    def __init__(self, files: Sequence[Path]) -> None:
        super().__init__()
        self.files: List[Path] = list(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = torch.load(self.files[idx])
        if "image" in item:
            return item["feat"], item["depth"], item["image"]
        return item["feat"], item["depth"]


@torch.inference_mode()
def prepare_feature_cache(
    model: nn.Module,
    loader: DataLoader,
    split: str,
    cache_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    rebuild: bool,
) -> List[Path]:
    """Precompute and store backbone features for a dataloader split."""

    dtype_name = dtype_name.lower()
    cache_root = cache_dir / split
    cache_root.mkdir(parents=True, exist_ok=True)
    meta_file = cache_root / "_meta.json"

    feature_files = sorted(cache_root.glob("*.pt"))
    if feature_files and meta_file.exists() and not rebuild:
        print(f"[cache] Using existing '{split}' cache from {cache_root}")
        return feature_files

    if rebuild:
        print(f"[cache] Rebuilding '{split}' cache at {cache_root}")
    else:
        print(f"[cache] Creating '{split}' cache at {cache_root}")

    for path in cache_root.glob("*.pt"):
        path.unlink()
    if meta_file.exists():
        meta_file.unlink()

    saved_paths: List[Path] = []
    feature_shape: Tuple[int, ...] | None = None
    model.eval()

    for batch_idx, (imgs, depths) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        depths = depths
        imgs_cpu = imgs.detach().cpu()

        feats_dict = model.encoder(imgs)
        feats = feats_dict["continuous"] if isinstance(feats_dict, dict) else feats_dict
        feats = torch.cat((feats[0], feats[1]), dim=1) if isinstance(feats, tuple) else feats
        feats = feats.detach().to(dtype=dtype).cpu()

        if feature_shape is None and feats.shape[0] > 0:
            feature_shape = tuple(feats[0].shape)

        for local_idx in range(feats.shape[0]):
            path = cache_root / f"{batch_idx:05d}_{local_idx:02d}.pt"
            torch.save({"feat": feats[local_idx], "depth": depths[local_idx], "image": imgs_cpu[local_idx]}, path)
            saved_paths.append(path)

    feature_files = sorted(saved_paths)
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump({"count": len(feature_files), "dtype": dtype_name, "feature_shape": list(feature_shape) if feature_shape else None}, f)

    return feature_files


class DepthProbe(nn.Module):
    """
    DINOv2-style depth probe (Lin-1):
      - frozen backbone features (B, C, H, W)
      - 1x1 conv -> logits over uniform depth bins (num_bins)
      - softmax + expectation over bin centers -> continuous depth
    Note: DINOv2 concatenates CLS to each patch token. Your model has no CLS,
    so we directly use the spatial features as-is.
    """
    def __init__(self, num_bins: int = 256, depth_min: float = 0.0, depth_max: float = 80.0, upsample_factor: int = 4) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.upsample_factor = int(upsample_factor)
        # Lazy so we don't need to know C at construction time
        #self.classifier = nn.LazyConv2d(self.num_bins, kernel_size=1, bias=True)
        self.classifier = nn.Sequential(
            nn.LazyConv2d(720, kernel_size=1, bias=True),           # 324 for 24-dim, 216 for 48-dim
            #nn.ReLU(inplace=True),
            nn.Conv2d(720, self.num_bins, kernel_size=1, bias=True)  # 384 -> num_classes
        )

        # Register bin centers as a buffer (will be moved with the module)
        edges = torch.linspace(self.depth_min, self.depth_max, self.num_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        self.register_buffer("bin_centers", centers, persistent=False)

    @torch.no_grad()
    def depth_to_bins(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous depth map (B,H,W) to bin indices (B,H,W).
        Invalid pixels should be handled by caller using IGNORE_INDEX.
        """
        # Clamp to range; caller should mask invalid (NaN/Inf) separately
        d = torch.clamp(depth, min=self.depth_min, max=self.depth_max)
        # Uniform bins
        bin_width = (self.depth_max - self.depth_min) / float(self.num_bins)
        idx = torch.floor((d - self.depth_min) / (bin_width + 1e-12)).long()
        idx = torch.clamp(idx, 0, self.num_bins - 1)
        return idx

    def forward(
        self,
        feats: torch.Tensor,
        target_size: Tuple[int, int] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          logits: (B, num_bins, H_out, W_out)
          depth:  (B, H_out, W_out) expected depth from softmax over bins
        """
        x = feats
        # DINOv2 depth probe upsamples by x4 (from patch grid) before comparing to GT
        if self.upsample_factor != 1:
            x = F.interpolate(x, scale_factor=self.upsample_factor, mode="bilinear", align_corners=False)

        logits = self.classifier(x)

        if target_size is not None and (logits.shape[-2:] != target_size):
            logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        probs = torch.softmax(logits, dim=1)
        centers = self.bin_centers.to(device=probs.device, dtype=probs.dtype).view(1, -1, 1, 1)
        depth = torch.sum(probs * centers, dim=1)
        return logits, depth


def compute_depth_metrics(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, float]:
    eps = 1e-6
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    if pred.numel() == 0:
        return {"absrel": float("nan"), "rmse": float("nan"), "rmse_log": float("nan"), "delta1": float("nan"), "n": 0}

    pred_clamped = torch.clamp(pred, min=eps)
    gt_clamped = torch.clamp(gt, min=eps)

    absrel = torch.mean(torch.abs(pred_clamped - gt_clamped) / gt_clamped).item()
    rmse = torch.sqrt(torch.mean((pred_clamped - gt_clamped) ** 2)).item()
    rmse_log = torch.sqrt(torch.mean((torch.log(pred_clamped) - torch.log(gt_clamped)) ** 2)).item()
    ratio = torch.max(pred_clamped / gt_clamped, gt_clamped / pred_clamped)
    delta1 = torch.mean((ratio < 1.25).float()).item()

    return {"absrel": absrel, "rmse": rmse, "rmse_log": rmse_log, "delta1": delta1, "n": int(pred.numel())}


def colorize_depth(
    depth: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    mode: str = "plasma",
) -> np.ndarray:
    depth_np = depth.copy()
    mask = np.isfinite(depth_np)
    if not np.any(mask):
        return np.zeros((*depth_np.shape, 3), dtype=np.uint8)

    if vmin is None or vmax is None:
        vmin, vmax = np.nanpercentile(depth_np, (2, 98))

    norm = np.zeros_like(depth_np, dtype=np.float32)
    norm[mask] = np.clip((depth_np[mask] - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    if mode == "white_to_black":
        cmap = plt.get_cmap("gray_r")
    elif mode == "plasma":
        cmap = plt.get_cmap("plasma")
    else:
        cmap = plt.get_cmap(mode)
    rgb = (255 * cmap(norm)[..., :3]).astype(np.uint8)
    rgb[~mask] = 0
    return rgb


# Main training/evaluation loop

def calculate_depth(args: argparse.Namespace, unknown_args: Sequence[str]) -> None:
    if args.seed > 0:
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.deterministic = True
        seed_everything(args.seed)

    device = torch.device(args.device)

    cfg_model = OmegaConf.load(args.config)
    cfg_model = OmegaConf.merge(cfg_model, OmegaConf.from_dotlist(list(unknown_args)))
    model = instantiate_from_config(cfg_model.model)
    state_dict = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    dataset_train = KITTIDepthEigen(
        root_dir=args.data_path,
        split_file=os.path.join(args.split_dir, "eigen_train_files_20pct.txt"),
        #split_file=os.path.join(args.split_dir, "multi_train_files.txt"),
        #split_file=os.path.join(args.split_dir, "sample_train_files.txt"),
        size=args.input_size,
    )
    dataset_val = KITTIDepthEigen(
        root_dir=args.data_path,
        split_file=os.path.join(args.split_dir, "eigen_val_files.txt"),
        #split_file=os.path.join(args.split_dir, "multi_val_files.txt"),
        #split_file=os.path.join(args.split_dir, "sample_val_files.txt"),
        size=args.input_size,
    )

    num_workers = min(4, os.cpu_count() or 1)
    use_cache = bool(args.cache_dir)
    cache_dir_path = Path(args.cache_dir) if use_cache else None
    cache_dtype = resolve_cache_dtype(args.cache_dtype) if use_cache else None

    if cache_dir_path is not None:
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        build_loader_kwargs = dict(
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        cache_train_loader = DataLoader(dataset_train, **build_loader_kwargs)
        cache_val_loader = DataLoader(dataset_val, **build_loader_kwargs)

        train_cache_files = prepare_feature_cache(
            model=model,
            loader=cache_train_loader,
            split="train",
            cache_dir=cache_dir_path,
            device=device,
            dtype=cache_dtype,
            dtype_name=args.cache_dtype,
            rebuild=args.rebuild_cache,
        )
        val_cache_files = prepare_feature_cache(
            model=model,
            loader=cache_val_loader,
            split="val",
            cache_dir=cache_dir_path,
            device=device,
            dtype=cache_dtype,
            dtype_name=args.cache_dtype,
            rebuild=args.rebuild_cache,
        )

        dataset_train = CachedFeatureDataset(train_cache_files)
        dataset_val = CachedFeatureDataset(val_cache_files)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=(device.type == "cuda"),
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=(device.type == "cuda"),
    )

    linear_probe = DepthProbe(num_bins=args.num_bins, depth_min=args.depth_min, depth_max=args.depth_max, upsample_factor=args.probe_upsample).to(device)
    opt = optim.AdamW(linear_probe.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    printed_classifier_params = False

    def run(loader: DataLoader, train: bool, cache_mode: bool, ep: int) -> float:
        nonlocal printed_classifier_params
        total_loss, total_count = 0.0, 0
        linear_probe.train(train)
        no_grad_ctx = torch.enable_grad() if train else torch.inference_mode()

        with no_grad_ctx:
            idx = 0
            metrics_acc = {"absrel": 0.0, "rmse": 0.0, "rmse_log": 0.0, "delta1": 0.0, "n": 0}

            for batch in loader:
                imgs_cpu = None
                if cache_mode:
                    if len(batch) == 3:
                        feats, depths, imgs_cpu = batch
                    else:
                        feats, depths = batch
                    param_dtype = next(linear_probe.classifier.parameters()).dtype
                    feats = feats.to(device, dtype=param_dtype, non_blocking=True)
                    depths = depths.to(device)
                    logits, pred = linear_probe(feats, target_size=depths.shape[-2:])
                else:
                    imgs, depths = batch
                    imgs = imgs.to(device, non_blocking=True)
                    imgs_cpu = imgs.detach().cpu()
                    depths = depths.to(device)
                    if args.num_input_frames > 1:
                        feats_dict = model.encoder(imgs, imgs)
                    else:
                        feats_dict = model.encoder(imgs)
                    feats = feats_dict["continuous"] if isinstance(feats_dict, dict) else feats_dict
                    feats = torch.cat(feats, dim=1) if isinstance(feats, tuple) else feats
                    logits, pred = linear_probe(feats.detach(), target_size=depths.shape[-2:])

                if not printed_classifier_params:
                    num_params = sum(p.numel() for p in linear_probe.classifier.parameters())
                    print(f"[info] depth classifier params: {num_params}")
                    printed_classifier_params = True

                valid_mask = torch.isfinite(depths)
                if valid_mask.sum() == 0:
                    continue

                # Discretize GT depth into uniform bins and train with cross-entropy on logits
                target_bins = torch.full_like(depths, fill_value=IGNORE_INDEX, dtype=torch.long)
                in_range = (depths >= args.depth_min) & (depths <= args.depth_max)
                use_mask = valid_mask & in_range
                if use_mask.sum() == 0:
                    continue
                target_bins[use_mask] = linear_probe.depth_to_bins(depths)[use_mask]
                loss = crit(logits, target_bins)
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                batch_size = depths.size(0)
                total_loss += float(loss.item()) * batch_size
                total_count += batch_size

                VIS_INDICES = [0, 1]
                vis_samples = [dataset_val[i] for i in VIS_INDICES]

                if not train:
                    batch_metrics = compute_depth_metrics(pred, depths, use_mask)
                    n = batch_metrics["n"]
                    if n > 0:
                        for k in metrics_acc.keys():
                            if k != "n":
                                metrics_acc[k] += batch_metrics[k] * n
                        metrics_acc["n"] += n

                    if args.dump_vis:
                        preds = pred.detach().cpu().numpy()
                        gts = depths.detach().cpu().numpy()
                        for b in range(batch_size):
                            gt_np = gts[b]
                            pd_np = preds[b]
                            finite_gt = np.isfinite(gt_np)
                            finite_pd = np.isfinite(pd_np)
                            combined = np.concatenate([gt_np[finite_gt], pd_np[finite_pd]])
                            if combined.size > 0:
                                vmin, vmax = np.nanpercentile(combined, (2, 98))
                            else:
                                vmin, vmax = None, None
                            gt_rgb = colorize_depth(gt_np, vmin=vmin, vmax=vmax, mode=args.depth_vis_mode)
                            pd_rgb = colorize_depth(pd_np, vmin=vmin, vmax=vmax, mode=args.depth_vis_mode)
                            if imgs_cpu is not None:
                                img = imgs_cpu[b].permute(1, 2, 0).numpy()
                                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                                img = np.clip(img * std + mean, 0.0, 1.0)
                                img_rgb = (255.0 * img).astype(np.uint8)
                                gt_rgb = (0.7 * gt_rgb + 0.3 * img_rgb).astype(np.uint8)
                                pd_rgb = (0.75 * pd_rgb + 0.25 * img_rgb).astype(np.uint8)
                            imageio.imwrite(f"{args.seq_real}/gt_{idx:05d}.png", gt_rgb)
                            imageio.imwrite(f"{args.seq_fake}/pred_{idx:05d}.png", pd_rgb)
                            idx += 1

                    if not train and idx == 0:
                        for vi, sample in enumerate(vis_samples):
                            if cache_mode and len(sample) == 3:
                                feat, depth_gt, img = sample
                                feat = feat.unsqueeze(0).to(device)
                                depth_gt = depth_gt.unsqueeze(0).to(device)
                                logits, depth_pred = linear_probe(feat, target_size=depth_gt.shape[-2:])
                                img_cpu = img
                            else:
                                img, depth_gt = sample
                                img = img.unsqueeze(0).to(device)
                                depth_gt = depth_gt.unsqueeze(0).to(device)
                                if args.num_input_frames > 1:
                                    feats = model.encoder(imgs, imgs)["continuous"]
                                else:
                                    feats = model.encoder(img)["continuous"]
                                logits, depth_pred = linear_probe(feats, target_size=depth_gt.shape[-2:])
                                img_cpu = img.cpu()[0]

                            # colorize
                            gt_rgb = colorize_depth(depth_gt[0].cpu().numpy())
                            pd_rgb = colorize_depth(depth_pred[0].detach().cpu().numpy())

                            writer.add_image(
                                f"depth/sample_{vi}/gt",
                                torch.from_numpy(gt_rgb).permute(2, 0, 1),
                                ep,
                            )
                            writer.add_image(
                                f"depth/sample_{vi}/pred",
                                torch.from_numpy(pd_rgb).permute(2, 0, 1),
                                ep,
                            )

            if not train and metrics_acc["n"] > 0:
                n_total = metrics_acc["n"]
                metrics = {
                    "AbsRel": metrics_acc['absrel']/n_total,
                    "RMSE": metrics_acc['rmse']/n_total,
                    "RMSE_log": metrics_acc['rmse_log']/n_total,
                    "δ1": metrics_acc['delta1']/n_total,
                }
            else:
                metrics = None
                # print(f"[eval] AbsRel: {metrics_acc['absrel']/n_total:.4f}  RMSE: {metrics_acc['rmse']/n_total:.3f}  RMSE_log: {metrics_acc['rmse_log']/n_total:.4f}  δ1: {metrics_acc['delta1']/n_total:.4f}")

        return total_loss / max(total_count, 1), metrics

    tb_dir = os.path.join(args.exp_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_dir)

    # log run info ONCE
    writer.add_text("run_info/num_epochs", str(args.num_epoch))
    writer.add_text("run_info/batch_size", str(args.batch_size))
    writer.add_text("run_info/lr", str(args.lr))

    for ep in range(1, args.num_epoch + 1):
        train_loss, _ = run(dataloader_train, True, use_cache, ep)
        writer.add_scalar("loss/train", train_loss, ep)

        if ep % args.eval_every == 0 or ep == args.num_epoch:
            val_loss, metrics = run(dataloader_val, False, use_cache, ep)
            writer.add_scalar("loss/val", val_loss, ep)
            if metrics:
                for k, v in metrics.items():
                    writer.add_scalar(f"metrics/{k}", v, ep)
        
    writer.close()

def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    val = v.lower()
    if val in {"yes", "true", "t", "y", "1"}:
        return True
    if val in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/eval a linear depth probe on KITTI Eigen split")
    parser.add_argument("--input_size", type=int, nargs=2, default=[256, 832])
    parser.add_argument("--num_input_frames", type=int, default=1)
    parser.add_argument("--num_bins", type=int, default=256)
    parser.add_argument("--depth_min", type=float, default=0.0)
    parser.add_argument("--depth_max", type=float, default=80.0)
    parser.add_argument("--probe_upsample", type=int, default=4)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.ckpt")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data_path", type=str, default="/data/nxtaimraid02/datasets/Kitti/")
    parser.add_argument("--split_dir", type=str, default="./evaluate/kitti_depth_splits/", help="Directory containing Eigen split txt files")
    parser.add_argument("--frames_dir", type=str, default="vis_depth_discrete")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--cache_dtype", type=str, default="float16")
    parser.add_argument("--rebuild_cache", type=str2bool, default=False)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--dump_vis", type=str2bool, default=False)
    parser.add_argument("--depth_vis_mode", type=str, default="plasma", choices=["turbo_r", "white_to_black", "plasma", "spring", "rainbow_r", "gist_ncar", "gist_rainbow_r"])

    args, unknown = parser.parse_known_args(argv)

    args.ckpt = os.path.join(args.exp_dir, args.ckpt)
    args.config = os.path.join(args.exp_dir, args.config)
    args.frames_dir = os.path.join(args.exp_dir, args.frames_dir)
    if args.cache_dir:
        args.cache_dir = os.path.join(args.exp_dir, args.cache_dir)

    args.seq_fake = os.path.join(args.frames_dir, "fake_images")
    args.seq_real = os.path.join(args.frames_dir, "real_images")
    return args, unknown


def main() -> None:
    args, unknown = parse_args()
    print(">>> Checkpoint:", args.ckpt)
    print(">>> Config:    ", args.config)
    print(">>> Data root: ", args.data_path)
    if args.cache_dir:
        print(">>> Cache dir: ", args.cache_dir)
        print(">>> Cache dtype:", args.cache_dtype)
        if args.rebuild_cache:
            print("[info] Cache rebuild requested.")

    if args.dump_vis:
        mkdir(args.seq_fake)
        mkdir(args.seq_real)
        print("[info] Visualization enabled- images may overwrite existing files.")

    calculate_depth(args=args, unknown_args=unknown)


if __name__ == "__main__":
    main()
