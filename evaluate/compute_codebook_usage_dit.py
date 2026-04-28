import sys
sys.path.append('../')
import os
import io
import argparse
import importlib
from einops import rearrange

import yaml
import random
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
# also disable grad to save memory
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import torch
torch.set_grad_enabled(False)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torchvision.utils import save_image
from torch.utils.data import Dataset, Subset
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import inception_v3

import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from models.first_stage.vqgan import VQModel
from models.second_stage.fm_model import Model

from pytorch_fid.fid_score import calculate_fid_given_paths
from compute_FID import calculate_fid, ImageFolderDataset, get_inception_features

try:
  sys.path.append('../Depth-Anything')
#   from depth_anything.dpt import DepthAnything
#   from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
except ImportError:
  print("Depth-Anything not found")

def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    val = v.lower()
    if val in {"yes", "true", "t", "y", "1"}:
        return True
    if val in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def load_gen(config, ckpt_path=None, is_gumbel=False):

  model = instantiate_from_config(config.model)

  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    # remove weights that contain "dino"
    sd = {k: v for k, v in sd.items() if "dino" not in k}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)
  return model.eval()

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def unnormalize_vqgan(x):
   B, C, H, W = x.shape
   x = x.reshape(B*C, H, W).permute(1, 2, 0)
   if isinstance(x, torch.Tensor):
     x = x.cpu().detach().numpy()
   image = ((x+1)*127.5).astype(np.uint8)
   if image.shape[-1] > 3:
      image = image[:, :, 3:]
   return image


def generate_with_dit(x, model):
    # could also use model(x) for reconstruction but use explicit encoding and decoding here
    
    # Set sample_with_ema=False for no ema_sampled images
    output = model.sample(x,eta=0.0, NFE=30, sample_with_ema=True, num_samples=5, frame_rate=None)[1]
    
    B, F, C, H, W = output.shape
    output = rearrange(output, 'B F C H W -> (B F) C H W', B=B, F=F)

    return output

def preprocess(img, target_image_size=256, map_dalle=True):
    img = PIL.Image.open(img)
    s = min(img.size)

    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')

    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    if map_dalle:
      img = map_pixels(img)
    return img

def generation_pipeline(model, image):
  generated = generate_with_dit(image, model)
  generated = generated[0].cpu().permute(1, 2, 0)
  generated = ((generated+1)*127.5).clamp_(0, 255).numpy().astype(np.uint8)
  return generated

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

def cluster_the_indices(indices, codes, max_d=1.8): # codes is (1, 256, 16, 16)
    unique_elements, inverse = np.unique(indices, return_inverse=True)
    indices = inverse
    codes = codes.squeeze(0).reshape(-1, codes.shape[2]*codes.shape[3]).transpose(0,1).cpu().numpy()
    distance_matrix = pdist(codes, metric='cosine')
    Z = linkage(distance_matrix, 'ward')
    max_d = max_d
    clusters = fcluster(Z, max_d, criterion='distance')
    # new index array with mapped clusters
    new_indices = np.zeros_like(indices)
    for i, c in enumerate(clusters):
        new_indices[indices==i] = c
    print(f'Number of clusters: {len(np.unique(clusters))}')
    return new_indices


def save_patches_by_index(image, indices, save_dir, idx, patch_size=16, upscale_factor=8):
    """
    Save patches by index.
    :param image: The input image.
    :param indices: The indices.
    :param patch_size: The size of the patch.
    :param save_dir: The root directory to save the patches.
    :param upscale_factor: The upscale factor.
    """
    unique_indices = np.unique(indices)
    original_size = image.shape[:2]
    upscaled_size = (original_size[0] * upscale_factor, original_size[1] * upscale_factor)
    image = Image.fromarray(image).resize(upscaled_size, resample=PIL.Image.NEAREST)
    image = np.asarray(image)

    save_dir = os.path.join(save_dir, 'patches')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for index in unique_indices:
        index_dir = os.path.join(save_dir, f'index_{index}')
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        
        # Find all patches corresponding to the current index
        positions = np.where(indices == index)
        for pos in zip(*positions):
            row = pos[0]//(original_size[0]//patch_size)
            col = pos[0]%(original_size[0]//patch_size) 
            start_x, start_y = row * patch_size*upscale_factor, col * patch_size*upscale_factor
            end_x, end_y = start_x + patch_size*upscale_factor, start_y + patch_size*upscale_factor
            patch = image[start_x:end_x, start_y:end_y, :]
            patch_img = Image.fromarray(patch)
            patch_img.save(os.path.join(index_dir, f'idx_{idx}_patch_{row}_{col}.png'))

def get_inception_model():
    model = inception_v3(pretrained=True)
    model.fc = torch.nn.Identity()  # Remove the classification layer
    model = model.to(DEVICE)
    return model

class ImageRFIDDataset(ImageFolderDataset):
    def __init__(self, folder_path, reconstructed_folder, transform=None):
        super().__init__(folder_path, transform)
        self.reconstructed_folder = reconstructed_folder

    def __getitem__(self, idx):
        orig_path = os.path.join(self.folder_path, self.image_filenames[idx])
        recons_path = os.path.join(self.reconstructed_folder, self.image_filenames[idx])
        orig_image = Image.open(orig_path).convert('RGB')
        recons_image = Image.open(recons_path).convert('RGB')
        if self.transform is not None:
            orig_image = self.transform(orig_image)
            recons_image = self.transform(recons_image)
        return orig_image, recons_image
       

def compute_gFID_score(model, path_original_images, path_gen_images):
    
    # Load the Inception model
    print(">> Loading Inception Model...")
    inception_model = get_inception_model()
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    original_images = ImageFolderDataset(folder_path=path_original_images, transform=transform)

    generated_images = ImageFolderDataset(folder_path=path_gen_images, transform=transform)
    generated_images.image_filenames = original_images.image_filenames  # let's make sure they match...
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if True:
      fid_score = calculate_fid_given_paths([path_original_images, path_gen_images], 16, DEVICE, 2048)
    else:
      print(">> Getting Inception Features...")
      original_features = get_inception_features(original_images, inception_model, DEVICE, transform)
      recontructed_features = get_inception_features(reconstructed_images, inception_model, DEVICE, transform)

      print(">> Calculating gID Score...")
      fid_score = calculate_fid(original_features, recontructed_features)

    print("FID Score:", fid_score)


def process_images(model, dataset, num_images):
    """
    Process the images in the input folder.
    :param input_folder: The folder containing the images.
    :param num_images: The number of images to process.
    :param codebook_size: The size of the codebook.
    """

    if args.compute_gFID_score:
      # prepare folders
      path_original_images = os.path.join(args.exp_dir, "original_images")
      path_gen_images = os.path.join(args.exp_dir, "generated_images")
      
      os.makedirs(path_original_images, exist_ok=True)
      os.makedirs(path_gen_images, exist_ok=True)
      # remove folder contents
      for f in os.listdir(path_original_images):
        os.remove(os.path.join(path_original_images, f))
      for f in os.listdir(path_gen_images):
        os.remove(os.path.join(path_gen_images, f))
     
      
    # Randomly select num_images from the list
    selected_images = random.sample(range(len(dataset)), num_images)


    # Process each selected image
    for idx, image_idx in enumerate(selected_images):
        if idx%100==0:
            print(f'{idx} images processed')
        # input_path = os.path.join(input_folder, image_name)
        
        sample = dataset[image_idx]

        image = sample["images"]

        # Modify the image
        generated = generation_pipeline(model, image=None)
        
          
        if args.compute_gFID_score:
          plt.imsave(os.path.join(path_original_images, f"image_{image_idx}.png"), unnormalize_vqgan(image))
          plt.imsave(os.path.join(path_gen_images, f"image_{image_idx}.png"), generated)
          
    if args.compute_gFID_score:      
      compute_gFID_score(model, path_original_images, path_gen_images)


def main(args):
  config = load_config(args.config_path, display=False) #99.85% zeros 
  model = load_gen(config, ckpt_path=args.ckpt_path).to(DEVICE)
  
  # dataset
  if args.data_config is not None:
    data_config = load_config(args.data_config, display=False)
    data = instantiate_from_config(data_config.data)
  else:
    data = instantiate_from_config(config.data)
  data.prepare_data()
  data.setup()
  data = data.datasets['validation']
    
  process_images(model, data, num_images=args.num_images)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
  parser.add_argument("--config_path", type=str, default=None, help="Path to the config file")
  parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the checkpoint file")

  # (optional) data config
  parser.add_argument("--data_config", type=str, default=None, help="Path to the data config file")

  parser.add_argument("--input_folder", type=str, default="./datasets/BDD100K/bdd100k/images/100k/test/", help="Path to input data folder")
  parser.add_argument("--create_index_visualization", action="store_true", help="Create index visualization")
  parser.add_argument("--cluster_indices", action="store_true", help="Cluster indices")
  parser.add_argument("--save_patches_by_index", action="store_true", help="Save patches by index")
  parser.add_argument("--compute_gFID_score", action="store_true", help="Compute gFID score")
  parser.add_argument("--num_images", type=int, default=1000, help="Number of images to process")
  parser.add_argument("--codebook_size", type=int, default=4096, help="Number of images to process")
  parser.add_argument("--num_input_frames", type=int, default=1, help="Number of input frames")
  parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
  parser.add_argument("--loss_supervision", type=str2bool, default=False)
  parser.add_argument("--auxiliary_depth", type=str2bool, default=False, help="For depth supervision loss model")
  parser.add_argument("--exp_dir", type=str, default="./visualizations_vae/default", help="Path to output data folder")
  

  args = parser.parse_args()
  
  if (args.config_path is None or args.ckpt_path is None):
     assert args.exp_name is not None, "Please provide the experiment name"
     args.config_path = os.path.join(os.environ['VQ_WORK_DIR'], args.exp_name, "config.yaml")
     args.ckpt_path = os.path.join(os.environ['VQ_WORK_DIR'], args.exp_name, "checkpoints", "last.ckpt")
     if not (os.path.exists(args.config_path) and os.path.exists(args.ckpt_path)):
        args.config_path = os.path.join("./logs", args.exp_name, "config.yaml")
        args.ckpt_path = os.path.join("./logs", args.exp_name, "checkpoints", "last.ckpt")

  try:
    # directory name of config_path
    args.exp_dir = os.path.join(os.environ['VQ_WORK_DIR'], 'visualizations', args.exp_name) #, os.path.basename(os.path.dirname(args.config_path)))
    if not os.path.exists(args.exp_dir):
      os.makedirs(args.exp_dir)
  except:
    args.exp_dir = './visualizations_vae/default'
  
  print(f"\n> Loading config from: {args.config_path}")
  print(f"> Loading checkpoint from: {args.ckpt_path}")
  print(f"> Saving visualizations to: {args.exp_dir}\n")

  # set seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  random.seed(args.seed)

  main(args)