import argparse
import os, glob
import random
# import math

# import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
from PIL import Image
from scipy.linalg import sqrtm
from tqdm import tqdm

import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.models import inception_v3

from omegaconf import OmegaConf
import sys
sys.path.append(".")
from main import instantiate_from_config

# Dataset class for loading generated images
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        self.image_filenames.sort()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_filenames[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

# Function to calculate the FID score
def calculate_fid(act1, act2):
    # Calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)

    # Calculate sqrt of product between covariances
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Function to get the features from images using Inception model
def get_inception_features(images, model, device, transform):
    model.eval()
    features = []

    with torch.no_grad():
        for img in tqdm(images):
            img = img.to(device).unsqueeze(0)
            feature = model(img)[0]
            features.append(feature.cpu().numpy())

    features = np.vstack(features)
    return features

# Main function
def main(args):
    # args = Args()

    args.exp_dir = os.path.join(args.work_dir, args.exp_name)
    os.makedirs(os.path.join(args.exp_dir, "FID", str(args.epoch), "reconstructed_images"), exist_ok=True)

    # Fixe seed
    if args.seed > 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.deterministic = True

    # Load pre-trained Inception model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inception_v3(pretrained=True)
    model.fc = torch.nn.Identity()  # Remove the classification layer
    model = model.to(device)

    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    labels, name = [1] * 1, "r_row"
    labels = torch.LongTensor(labels).to(device)
    fid_score = []

    real_images = ImageFolderDataset(folder_path=args.data_folder, transform=transform) # Your dataset of real images
    real_images = Subset(real_images, np.random.choice(len(real_images), args.num_samples, replace=False))

    vqgan_cfg_path = [p for p in os.listdir(os.path.join(args.exp_dir, "configs")) if p.endswith("project.yaml")][0]
    vqgan_ckpt_path = os.path.join(args.exp_dir, "checkpoints", f"checkpoint-epoch={args.epoch}.ckpt")
    vqgan = instantiate_from_config(OmegaConf.load(os.path.join(args.exp_dir, "configs", vqgan_cfg_path)).model).eval().cuda()
    vqgan.load_state_dict(torch.load(vqgan_ckpt_path)["state_dict"])
    print(">> Loaded VQGAN ckpt from:", vqgan_ckpt_path)

    # Reconstruct N samples for FID score calculation
    for i in tqdm(range(args.num_samples)):
        reconst_sample = vqgan(real_images[i].unsqueeze(0).cuda())[0].cpu()
        save_image(reconst_sample, os.path.join(args.exp_dir, "FID", str(args.epoch), f"reconstructed_images/index_{i}.jpg"))

    # Load your datasets here
    reconst_images = ImageFolderDataset(folder_path=os.path.join(args.exp_dir, "FID", str(args.epoch), f"reconstructed_images/"), transform=transform) # Your dataset of generated images

    # Get features from the Inception model
    print(">> Getting features for real images...")
    real_features = get_inception_features(real_images, model, device, transform)
    print(">> Getting features for reconstructed images...")
    gen_features = get_inception_features(reconst_images, model, device, transform)

    # Calculate FID score
    print(">> Calculating FID Score...")
    fid_score = calculate_fid(real_features, gen_features)
    print("FID Score:", fid_score)

    # # Write the FID score to a txt file
    with open(os.path.join(args.exp_dir, "FID", str(args.epoch), "FID_Score.txt"), "w") as f:
        f.write(str(fid_score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, help="Name of the experiment")
    parser.add_argument("epoch", type=int, help="Epoch number")
    parser.add_argument("--data_folder", type=str, default="/datasets/BDD100K/bdd100k/images/100k/val", help="Path to the dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of real/produced samples for evaluation")

    args = parser.parse_args()

    args.work_dir = os.environ.get("VQ_WORK_DIR", ".")

    main(args)