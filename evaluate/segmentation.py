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
from torchmetrics.classification import JaccardIndex
from torchvision.datasets import Cityscapes
from torch.utils.tensorboard import SummaryWriter

# project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from util import instantiate_from_config  # noqa: E402

# Constants / metadata

IGNORE_INDEX: int = 255

# Cityscapes IDs: https://www.cityscapes-dataset.com/dataset-overview/#fine-annotation-labels
# We keep IGNORE_INDEX as 255 and only map train IDs for valid classes (19 classes).
VALID_LABEL_IDS: List[int] = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

CLASS_NAMES: List[str] = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
    "traffic_sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]
N_CLASSES: int = len(VALID_LABEL_IDS)

PALETTE: np.ndarray = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ],
    dtype=np.uint8,
)

# Utilities

def get_ckpt_epoch_step(ckpt_path: str) -> Tuple[int, int]:
    """Return (epoch, global_step) stored inside a Lightning checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return int(ckpt.get("epoch", -1)), int(ckpt.get("global_step", -1))


def mkdir(path: os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def build_id_mapping() -> torch.Tensor:
    """
    Build a vectorized mapping from original Cityscapes label IDs -> train IDs.
    - valid labels -> 0..18 (order matches VALID_LABEL_IDS)
    - everything else -> IGNORE_INDEX
    Returns a tensor of length 256 to use as a lookup table.
    """
    lut = torch.full((256,), IGNORE_INDEX, dtype=torch.long)
    for train_id, label_id in enumerate(VALID_LABEL_IDS):
        lut[label_id] = train_id
    return lut


ID_LUT: torch.Tensor = build_id_mapping()  # CPU LUT (moved to device at use time)


def encode_segmap(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert Cityscapes label IDs to train IDs.
    Keeps IGNORE_INDEX pixels as IGNORE_INDEX.
    Args:
        mask: (B,H,W) or (H,W) tensor of original label IDs.
    """
    if mask.dtype != torch.long:
        mask = mask.long()
    lut = ID_LUT.to(mask.device)
    # Values outside [0,255] are clamped to 255 and thus mapped to IGNORE_INDEX
    mask = mask.clamp_min(0).clamp_max(255)
    mapped = lut[mask]
    return mapped


def colorize(labels: torch.Tensor) -> np.ndarray:
    """
    Convert train-ID label map to RGB (uint8) using PALETTE.
    - labels: (H,W) long tensor with values in {0..N_CLASSES-1} or IGNORE_INDEX.
    IGNORE_INDEX pixels will be colored black.
    """
    arr = labels.detach().cpu().numpy()
    h, w = arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    valid = arr != IGNORE_INDEX
    rgb[valid] = PALETTE[arr[valid]]
    return rgb


# Dataset

class CityScapes(Cityscapes):
    """Cityscapes wrapper with Albumentations transform producing tensors."""

    def __init__(self, size: Tuple[int, int], data_path, *args, **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.transform = A.Compose(
            [
                A.Resize(size[0], size[1]),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # scale to [-1, 1]
                ToTensorV2(),
            ]
        )


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[index]).convert("RGB")

        # Cityscapes target loading (supports 'semantic', 'polygon', etc.)
        targets: List[Any] = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]

        transformed = self.transform(image=np.array(image), mask=np.array(target))
        img_t: torch.Tensor = transformed["image"]  # (C,H,W), float32
        mask_t: torch.Tensor = transformed["mask"]  # (H,W), int/uint

        return img_t, torch.as_tensor(mask_t)


CACHE_DTYPE_MAP: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_cache_dtype(name: str) -> torch.dtype:
    key = name.lower()
    if key not in CACHE_DTYPE_MAP:
        valid = ", ".join(sorted(CACHE_DTYPE_MAP))
        raise ValueError(f"Unsupported cache dtype '{name}'. Valid options: {valid}")
    return CACHE_DTYPE_MAP[key]


class CachedFeatureDataset(Dataset):
    """Dataset that serves cached feature tensors and encoded labels."""

    def __init__(self, files: Sequence[Path]) -> None:
        super().__init__()
        self.files: List[Path] = list(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = torch.load(self.files[idx])
        return item["feat"], item["label"]


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

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels_encoded = encode_segmap(labels).cpu()

        feats_dict = model.encode(imgs)
        feats = feats_dict["continuous"] if isinstance(feats_dict, dict) else feats_dict
        
        feats = torch.cat((feats[0], feats[1]), dim=1) if isinstance(feats, tuple) else feats
        
        feats = feats.detach().to(dtype=dtype).cpu()
        
        if feature_shape is None and feats.shape[0] > 0:
            feature_shape = tuple(feats[0].shape)

        for local_idx in range(feats.shape[0]):
            path = cache_root / f"{batch_idx:05d}_{local_idx:02d}.pt"
            torch.save({"feat": feats[local_idx], "label": labels_encoded[local_idx]}, path)
            saved_paths.append(path)

    feature_files = sorted(saved_paths)
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "count": len(feature_files),
                "dtype": dtype_name,
                "feature_shape": list(feature_shape) if feature_shape else None,
            },
            f,
        )

    return feature_files


# Model heads

class SegProbe(nn.Module):
    """1×1 conv classifier head for dense features."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.LazyConv2d(num_classes, kernel_size=1, bias=True)

    def forward(self, feats: torch.Tensor, target_size: Tuple[int, int] | None = None) -> torch.Tensor:
        logits_lr = self.classifier(feats)  # (B,C,h,w)
        if target_size is None:
            return logits_lr
        return F.interpolate(logits_lr, size=target_size, mode="bilinear", align_corners=False)


# Train / Eval

def calculate_semantic(args: argparse.Namespace, unknown_args: Sequence[str]) -> None:
    """Train a linear probe and optionally dump colored predictions/GT."""

    # Determinism (optional)
    if args.seed > 0:
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.deterministic = True
        seed_everything(args.seed)

    device = torch.device(args.device)
    tb_dir = os.path.join(args.exp_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_dir)

    writer.add_text("run_info/num_epochs", str(args.num_epoch))

    # Load backbone from Hydra config
    cfg_model = OmegaConf.load(args.config)
    cfg_model = OmegaConf.merge(cfg_model, OmegaConf.from_dotlist(list(unknown_args)))
    model = instantiate_from_config(cfg_model.model)
    state_dict = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Metrics
    perclass_metric = JaccardIndex(
        task="multiclass",
        num_classes=N_CLASSES,
        ignore_index=IGNORE_INDEX,
        average="none",
    ).to(device)

    miou_metric = JaccardIndex(
        task="multiclass",
        num_classes=N_CLASSES,
        ignore_index=IGNORE_INDEX,
        average="macro",
    ).to(device)

    # Datasets / loaders
    dataset_train = CityScapes(size=args.input_size,
        data_path=args.data_path, split="train", mode="fine", target_type="semantic"
    )
    dataset_val = CityScapes(
        size=args.input_size,
        data_path=args.data_path, split="val", mode="fine", target_type="semantic"
    )

    # Pick fixed samples for visualization
    VIS_INDICES = [0, 1]  # choose any 1–2 indices
    vis_samples = [dataset_val[i] for i in VIS_INDICES]

    num_workers = min(4, os.cpu_count() or 1)
    use_cache = bool(args.cache_dir)
    cache_dir_path = Path(args.cache_dir) if use_cache else None
    cache_dtype = resolve_cache_dtype(args.cache_dtype) if use_cache else None

    if cache_dir_path is not None:
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        loader_kwargs = dict(
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        cache_train_loader = DataLoader(dataset_train, **loader_kwargs)
        cache_val_loader = DataLoader(dataset_val, **loader_kwargs)

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

    # Linear probe head
    linear_probe = SegProbe(N_CLASSES).to(device)
    opt = optim.AdamW(linear_probe.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def unpack_sample(sample):
        if use_cache:
            feats, label = sample
            return feats.unsqueeze(0), label.unsqueeze(0)
        else:
            img, label = sample
            return img.unsqueeze(0), label.unsqueeze(0)

    def log_segmentation_images(
        writer,
        linear_probe,
        model,
        samples,
        epoch,
        device,
    ):
        linear_probe.eval()
        with torch.inference_mode():
            for i, sample in enumerate(samples):
                x, label = unpack_sample(sample)
                x = x.to(device)
                label = label.to(device)

                if use_cache:
                    logits = linear_probe(x, target_size=label.shape[-2:])
                else:
                    feats = model.encoder(x)
                    # feats = feats["continuous"] if isinstance(feats, dict) else feats
                    feats = torch.cat((feats[0], feats[1]), dim=1) if isinstance(feats, tuple) else feats
                    logits = linear_probe(feats, target_size=label.shape[-2:])

                pred = logits.argmax(dim=1)[0]

                gt_rgb = colorize(label[0])
                pred_rgb = colorize(pred)

                # HWC → CHW and normalize to [0,1]
                gt_rgb = torch.from_numpy(gt_rgb).permute(2, 0, 1) / 255.0
                pred_rgb = torch.from_numpy(pred_rgb).permute(2, 0, 1) / 255.0

                writer.add_image(f"segmentation/sample_{i}/gt", gt_rgb, epoch)
                writer.add_image(f"segmentation/sample_{i}/pred", pred_rgb, epoch)

    def run(loader: DataLoader, train: bool, cache_mode: bool) -> float:
        total_loss, total_count = 0.0, 0
        linear_probe.train(train)

        perclass_metric.reset()
        miou_metric.reset()

        # use inference_mode for eval to disable autograd & cudnn benchmarking side-effects
        no_grad_ctx = torch.enable_grad() if train else torch.inference_mode()
        with no_grad_ctx:
            idx = 0
            for batch in loader:
                if cache_mode:
                    feats, labels = batch
                    feats = feats.to(
                        device=device,
                        dtype=linear_probe.classifier.weight.dtype,
                        non_blocking=True,
                    )
                    labels = labels.to(device)
                    logits = linear_probe(feats, target_size=labels.shape[-2:])
                else:
                    imgs, labels = batch
                    imgs = imgs.to(device, non_blocking=True)
                    labels = labels.to(device)

                    # map original Cityscapes IDs -> train IDs, keep IGNORE_INDEX intact
                    labels = encode_segmap(labels)

                    feats = model.encoder(imgs)
                    # print("Shape of feats : ", feats.shape)
                    # feats = feats_dict["continuous"] if isinstance(feats_dict, dict) else feats_dict  # (B,C,h,w)
                            
                    feats = torch.cat((feats[0], feats[1]), dim=1) if isinstance(feats, tuple) else feats

                    logits = linear_probe(feats.detach(), target_size=labels.shape[-2:])
                loss = crit(logits, labels)

                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_count += batch_size

                if not train:
                    miou_metric.update(logits, labels)
                    perclass_metric.update(logits, labels)

                    # optional visualization dump
                    if args.dump_vis:
                        preds = logits.argmax(dim=1)  # (B,H,W)
                        for b in range(batch_size):
                            gt_rgb = colorize(labels[b])
                            pd_rgb = colorize(preds[b])
                            imageio.imwrite(f"{args.seq_real}/gt_{idx:05d}.png", gt_rgb)
                            imageio.imwrite(f"{args.seq_fake}/pred_{idx:05d}.png", pd_rgb)
                            idx += 1

        if not train:
            miou = float(miou_metric.compute().item())
            # ious: List[float] = list(map(float, perclass_metric.compute().detach().cpu().tolist()))
            per_class_ious = perclass_metric.compute().detach().cpu().tolist()
            print(f"[eval] mIoU: {miou:.4f}")
            print(f"[eval] per-class IoU: {per_class_ious}")
            return total_loss / max(total_count, 1), miou, per_class_ious

        return total_loss / max(total_count, 1), None, None

    for ep in range(1, args.num_epoch + 1):
        train_loss, _, _ = run(dataloader_train, train=True, cache_mode=use_cache)
        if ep % args.eval_every == 0 or ep == args.num_epoch:
            val_loss, miou, per_class_ious = run(
                dataloader_val, train=False, cache_mode=use_cache
            )
                # ---- TensorBoard logging ----
            writer.add_scalar("loss/train", train_loss, ep)
            writer.add_scalar("loss/val", val_loss, ep)
            writer.add_scalar("metrics/mIoU", miou, ep)

            log_segmentation_images(
                writer=writer,
                linear_probe=linear_probe,
                model=model,
                samples=vis_samples,
                epoch=ep,
                device=device,
            )

            for cls_name, cls_iou in zip(CLASS_NAMES, per_class_ious):
                writer.add_scalar(f"metrics_per_class/{cls_name}", cls_iou, ep)

            print(
                f"epoch {ep:03d}: train {train_loss:.3f}  "
                f"val {val_loss:.3f}  mIoU {miou:.4f}"
            )
        else:
            writer.add_scalar("loss/train", train_loss, ep)
            print(f"epoch {ep:03d}: train {train_loss:.3f}")
        
        # #count number of parameters in linear probe
        # n_params = sum(p.numel() for p in linear_probe.parameters() if p.requires_grad)
        # print(f"[train] linear probe parameters: {n_params}")

    writer.close()

# CLI

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
    parser = argparse.ArgumentParser(description="Train/eval a linear segmentation probe on Cityscapes features.")
    parser.add_argument("--input_size", type=int, nargs=2, default=[288, 512], help="Resize (H W) for Cityscapes images")

    # Experiment paths
    parser.add_argument("--exp_dir", type=str, required=True, help="Root experiment directory")
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.ckpt", help="Checkpoint path (relative to exp_dir)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Model config (relative to exp_dir)")
    parser.add_argument("--data_path", type=str, default="/data/lmbraid12/datasets/public/Cityscapes/", help="Path to the Cityscapes dataset root")
    parser.add_argument("--frames_dir", type=str, default="vis_semantic", help="Where to dump visualizations (relative to exp_dir)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional directory to cache backbone features (relative to exp_dir)")
    parser.add_argument("--cache_dtype", type=str, default="float16", help="Storage dtype for cached features")
    parser.add_argument("--rebuild_cache", type=str2bool, default=False, help="Force rebuilding cached features even if present")
    # Optimization
    parser.add_argument("--num_epoch", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Global batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--eval_every", type=int, default=5, help="Evaluate every N epochs")
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed (<=0 disables repeatability)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="cpu | cuda")
    parser.add_argument("--dump_vis", type=str2bool, default=True, help="Dump colored GT/pred PNGs during eval")

    args, unknown = parser.parse_known_args(argv)

    # Expand checkpoint/config/frames paths relative to exp_dir
    args.ckpt = os.path.join(args.exp_dir, args.ckpt)
    args.config = os.path.join(args.exp_dir, args.config)
    args.frames_dir = os.path.join(args.exp_dir, args.frames_dir)
    if args.cache_dir:
        args.cache_dir = os.path.join(args.exp_dir, args.cache_dir)

    # Output dirs (created later if dump_vis)
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

    # Prepare output dirs if we're dumping
    if args.dump_vis:
        mkdir(args.seq_fake)
        mkdir(args.seq_real)
        print("[info] Visualization enabled – images may overwrite existing files.")

    calculate_semantic(args=args, unknown_args=unknown)


if __name__ == "__main__":
    main()