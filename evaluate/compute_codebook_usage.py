import sys
sys.path.append('../')
import os
import io
import argparse
import importlib

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

from pytorch_fid.fid_score import calculate_fid_given_paths
from compute_FID import calculate_fid, ImageFolderDataset, get_inception_features

try:
  sys.path.append('../Depth-Anything')
#   from depth_anything.dpt import DepthAnything
#   from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
except ImportError:
  print("Depth-Anything not found")

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

def load_vqgan(config, ckpt_path=None, is_gumbel=False):

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
   return image

def reconstruct_with_vqgan(x, model, hist, codebook_size):
    # could also use model(x) for reconstruction but use explicit encoding and decoding here
    if args.num_input_frames > 1:
        output = model.encode(x, x)
    else:
        output = model.encode(x)
    codes = output["quantized"]
    indices = output["indices"]
    # print("Shape of indices : ", indices.shape)
    if "vqgan_hierarchical_multires" in str(model.__class__):
      codes = codes[1]; indices = indices[1][-1]
      reconst_sample = model.decode(codes, i=1)
    else:
      indices = indices[-1]
      reconst_sample = model.decode(codes)
    if isinstance(reconst_sample, tuple):
        reconst_sample = reconst_sample[0]
    if isinstance(reconst_sample, tuple):
        reconst_sample = reconst_sample[0]
    #print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
    if len(indices.shape) > 1:
      indices = indices.view(-1)
    indices = indices.flatten()
    for index in indices.tolist():
        if 0 <= index < codebook_size:
            hist[int(index)] += 1
    return reconst_sample, hist, indices, codes

def create_histogram(codebook_size):
    """Create a histogram with 1025 bins (for indices 0 to 1024)."""
    # Initialize histogram with zeros
    return [0] * codebook_size

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

def reconstruction_pipeline(model, image, hist, size, codebook_size):
  if len(image.shape) == 3:
    x_vqgan = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
  else:
    x_vqgan = image.to(DEVICE)
  reconstructed, hist, indices, codes = reconstruct_with_vqgan(x_vqgan, model, hist, codebook_size)
  reconstructed = reconstructed[0].cpu().permute(1, 2, 0)
  reconstructed = ((reconstructed+1)*127.5).clamp_(0, 255).numpy().astype(np.uint8)
  return reconstructed, hist, indices, x_vqgan, codes

def plot_histogram(hist, exp_dir, save_path = 'histogram.png'):
    """
    Plot the histogram.
    :param hist: The histogram to plot.
    """
    plt.figure(figsize=(10, 5), dpi=1000)
    plt.bar(range(len(hist)), hist)
    plt.title('Histogram of Indices')
    plt.xlabel('Index')
    plt.ylabel('Frequency')
    hist_path = os.path.join(exp_dir, save_path)
    plt.savefig(hist_path)

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

def create_index_visualization(img, indices, codes, patch_size, colors, save_dir, upscale_factor=4, idx=0):
    """
    Create a visualization of the indices.
    :param img: The input image.
    :param indices: The indices.
    :param patch_size: The size of the patch.
    :return: The visualization.
    """
    # convert shape from (3, 224, 224) to (224, 224, 3)
    # img = (img * 255).astype(np.uint8)
    # img = np.transpose(img, (1, 2, 0))
    original_size = img.shape[:2]
    upscaled_size = (original_size[0] * upscale_factor, original_size[1] * upscale_factor)
    pil_img = Image.fromarray(img).resize(upscaled_size, resample=PIL.Image.NEAREST)
    draw = ImageDraw.Draw(pil_img)
    
    indices = indices.reshape(1, 16, 16).cpu().numpy()
    # Attempt to use a default font, or fall back to PIL's default if not specified
    try:
        font = ImageFont.truetype("arial.ttf", 2)
    except IOError:
        font = ImageFont.load_default()

    for i in range(indices.shape[1]): # 14
        for j in range(indices.shape[2]): # 14
            index = indices[0, i, j]
            color = colors[index]
            # Define the patch area
            start_x, start_y = i * patch_size*upscale_factor, j * patch_size*upscale_factor
            end_x, end_y = start_x + patch_size*upscale_factor, start_y + patch_size*upscale_factor
            # Overlay color with transparency
            overlay = (0.8 * np.array(pil_img.crop((start_y, start_x, end_y, end_x))) + 0.2 * color).astype(np.uint8)
            pil_img.paste(Image.fromarray(overlay), (start_y, start_x))
            # Draw index number
            draw.text((start_y + 5, start_x + 5), str(index), fill="black", font=font)

    img_vis = np.array(pil_img)
    img_vis = Image.fromarray(img_vis.astype(np.uint8))
    img_vis_path_dir = os.path.join(save_dir, 'semantic_mapping')
    if not os.path.exists(img_vis_path_dir):
      os.makedirs(img_vis_path_dir)
    img_vis_path = os.path.join(img_vis_path_dir, f'index_{idx}.png')
    img_vis.save(img_vis_path)

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
       

def compute_rFID_score(model, path_original_images, path_recons_images):
    
    # Load the Inception model
    print(">> Loading Inception Model...")
    inception_model = get_inception_model()
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    original_images = ImageFolderDataset(folder_path=path_original_images, transform=transform)
    # get reconstructed images from the VQGAN model

    reconstructed_images = ImageFolderDataset(folder_path=path_recons_images, transform=transform)
    reconstructed_images.image_filenames = original_images.image_filenames  # let's make sure they match...
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if True:
      fid_score = calculate_fid_given_paths([path_original_images, path_recons_images], 16, DEVICE, 2048)
    else:
      print(">> Getting Inception Features...")
      original_features = get_inception_features(original_images, inception_model, DEVICE, transform)
      recontructed_features = get_inception_features(reconstructed_images, inception_model, DEVICE, transform)

      print(">> Calculating FID Score...")
      fid_score = calculate_fid(original_features, recontructed_features)

    print("FID Score:", fid_score)


def process_images(model, dataset, num_images, codebook_size, exp_dir):
    """
    Process the images in the input folder.
    :param input_folder: The folder containing the images.
    :param num_images: The number of images to process.
    :param codebook_size: The size of the codebook.
    """

    if args.compute_rFID_score:
      # prepare folders
      path_original_images = os.path.join(args.exp_dir, "original_images")
      path_recons_images = os.path.join(args.exp_dir, "reconstructed_images")
      
      os.makedirs(path_original_images, exist_ok=True)
      os.makedirs(path_recons_images, exist_ok=True)
      # remove folder contents
      for f in os.listdir(path_original_images):
        os.remove(os.path.join(path_original_images, f))
      for f in os.listdir(path_recons_images):
        os.remove(os.path.join(path_recons_images, f))
     
      # compute_rFID_score(model, path_original_images, num_images, exp_dir)

    # Create a histogram with 1025 bins (for indices 0 to 1024):
    histo = create_histogram(codebook_size)

    # Get all image files from the input folder
    # all_images = [file for file in os.listdir(input_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Randomly select num_images from the list
    selected_images = random.sample(range(len(dataset)), num_images)

     # Generate random colors for each index
    colors = np.random.randint(0, 255, (codebook_size, 3))

    # Process each selected image
    for idx, image_idx in enumerate(selected_images):
        if idx%100==0:
            print(f'{idx} images processed')
        # input_path = os.path.join(input_folder, image_name)
        
        sample = dataset[image_idx]
        #image = sample.unsqueeze(0)
        #input_path = sample['file_path_']
        image = sample['images']

        # Modify the image
        reconstructed, histo, indices, x_vqgan, codes = reconstruction_pipeline(model, image, hist=histo, size=256, codebook_size=codebook_size)
        

        if args.cluster_indices:
           indices = cluster_the_indices(indices.cpu().numpy(), codes, max_d=1.9)
           indices = torch.from_numpy(indices).cuda() 

        # create visualization of indices
        if args.create_index_visualization:
          create_index_visualization(img=unnormalize_vqgan(image),
                                    codes=codes, 
                                    indices=indices, 
                                    patch_size=16, 
                                    colors=colors,
                                    upscale_factor=4, 
                                    save_dir=exp_dir, 
                                    idx=image_idx)

        #save patches by index
        if args.save_patches_by_index:
          save_patches_by_index(image=unnormalize_vqgan(image),
                              indices=indices.cpu().numpy(),
                              save_dir=exp_dir,
                              idx=image_idx)
          
        if args.compute_rFID_score:
          # save original images for rFID score TODO use orig image names
          # save_image(x_vqgan[0], os.path.join(path_original_images, f"original_image_{image_idx}.jpg"))
          plt.imsave(os.path.join(path_original_images, f"image_{image_idx}.png"), unnormalize_vqgan(image))
          plt.imsave(os.path.join(path_recons_images, f"image_{image_idx}.png"), reconstructed)
          
    if args.compute_rFID_score:      
      compute_rFID_score(model, path_original_images, path_recons_images)

    #plot_histogram(histo)
    print(100*sum([int(h!=0) for h in histo]) / len(histo), '% codebook usage')

def main(args):
  config = load_config(args.config_path, display=False) #99.85% zeros 
  model = load_vqgan(config, ckpt_path=args.ckpt_path).to(DEVICE)
  try:
    codebook_size = config.model.params.quantizer_config.params.get("n_e", args.codebook_size)
  except:
    codebook_size = config.model.params.get("n_embed", args.codebook_size)
  
  # dataset
  if args.data_config is not None:
    data_config = load_config(args.data_config, display=False)
    data = instantiate_from_config(data_config.data)
  else:
    data = instantiate_from_config(config.data)
  data.prepare_data()
  data.setup()
  data = data.datasets['validation']
    
  process_images(model, data, num_images=args.num_images, codebook_size=codebook_size, exp_dir=args.exp_dir)

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
  parser.add_argument("--compute_rFID_score", action="store_true", help="Compute rFID score")
  parser.add_argument("--num_images", type=int, default=1000, help="Number of images to process")
  parser.add_argument("--codebook_size", type=int, default=1024, help="Number of images to process")
  parser.add_argument("--num_input_frames", type=int, default=1, help="Number of input frames")
  parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
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
    args.exp_dir = './visualizations/default'
  
  print(f"\n> Loading config from: {args.config_path}")
  print(f"> Loading checkpoint from: {args.ckpt_path}")
  print(f"> Saving visualizations to: {args.exp_dir}\n")

  # set seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  random.seed(args.seed)

  main(args)