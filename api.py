"""
A Python API for the EMDiffuse Super-Resolution model that processes data
 directly as input and provides output without relying on file I/O.
"""
import os
import urllib
import zipfile
import core.praser as Praser
from models import create_EMDiffuse
from emdiffuse_conifg import EMDiffuseConfig
import core.util as Util
import torch
from torchvision import transforms
import contextlib

def download_weights():
    # todo remove the hard coded path and other parameters
    if not os.path.isdir('./experiments'):
        os.mkdir('./experiments')
    zipPath="experiments/EMDiffuse-r.zip"
    if not os.path.exists(zipPath):
        data = urllib.request.urlretrieve('https://zenodo.org/records/10686160/files/EMDiffuse-r.zip?download=1', zipPath)
        with zipfile.ZipFile(zipPath, 'r') as zip_ref:
            zip_ref.extractall("experiments")


def create_model():
    download_weights()
    # in order to use the original code, we need to create a dummy directory
    dummy_dir = './dummy_dir'
    if not os.path.exists(dummy_dir):
        os.makedirs(dummy_dir)
    
    config = EMDiffuseConfig(config='config/EMDiffuse-r.json', path=dummy_dir, batch_size=1, phase='test', mean=1, resume='./experiments/EMDiffuse-r/best', step=200)
    opt = Praser.parse(config)
    opt['world_size'] = 1
    opt['gpu_ids'] = [0]
    opt['seed'] = 1
    Util.set_seed(opt['seed'])
    model = create_EMDiffuse(opt)
    return model


model = create_model()


def super_resolution(input_image, scale_factor):
    """
    Args:
        input_image: torch tensor of shape (1, 1, H, W) in the range [0, 1] float32 on GPU
        scale_factor: the factor by which the image should be upscaled
    """
    with contextlib.redirect_stdout(None):
      h, w = input_image.shape[-2:]
      image_size = (h * scale_factor, w * scale_factor)
      model.batch_size = 1
      model.tfs = transforms.Compose([
              transforms.Resize(image_size),
              transforms.Normalize(mean=[0.5], std=[0.5])
          ])
      
      model.netG.eval()
      with torch.no_grad():
          model.cond_image = model.tfs(input_image)
          model.output = model.model_test(model.sample_num)
          return model.output
