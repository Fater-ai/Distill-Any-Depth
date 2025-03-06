# predict.py
from typing import Any
from cog import BasePredictor, Input, Path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add repo root to path for imports

import os.path as osp
import numpy as np
import torch
from PIL import Image
import cv2
from torchvision.transforms import Compose
from distillanydepth.midas.transforms import Resize, NormalizeImage, PrepareForNet
from distillanydepth.modeling.archs.dam.dam import DepthAnything
from distillanydepth.depth_anything_v2.dpt import DepthAnythingV2
from distillanydepth.utils.image_util import chw2hwc, colorize_depth_maps
from safetensors.torch import load_file

class Predictor(BasePredictor):
    def setup(self):
        """Load the DepthAnything model into memory."""
        print("Loading Distilled Large Depth Anything V1 model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Determine device in setup
        model_kwargs = dict(
            vitl=dict( # Using vitl config as per 'depthanything-large' in infer.py
                encoder="vitl",
                features=256,
                out_channels=[256, 512, 1024, 1024],
                use_bn=False,
                use_clstoken=False,
                max_depth=150.0,
                mode='disparity',
                pretrain_type='dinov2',
                del_mask_token=False
            )
        )

        self.model = DepthAnything(**model_kwargs['vitl']).to(self.device)
        self.model.mode
        model_weights = load_file("checkpoint/large/model.safetensors")
        self.model.load_state_dict(model_weights)
        del model_weights
        torch.cuda.empty_cache()
        self.model.eval()
        print("Distilled Large Depth Anything V1 model loaded.")
 
    def predict(self,
            image: Path = Input(description="Input image for depth estimation"),
            ) -> Path:
        """Run depth estimation and return path to depth map JPG."""

        print(f"Processing image: {image}")

        transform = Compose([
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet()
        ])

        image_path = str(image) # Convert cog.Path to string
        image_np = cv2.imread(image_path, cv2.COLOR_BGR2RGB)[..., ::-1] / 255
        normalized_image = transform({'image': image_np})['image']
        normalized_image = torch.from_numpy(normalized_image).unsqueeze(0).to(self.device)

        with torch.autocast("cuda"):
            pred_disp, _ = self.model(normalized_image)
        pred_disp_np = pred_disp.cpu().detach().numpy()[0, :, :, :].transpose(1, 2, 0)
        pred_disp = (pred_disp_np - pred_disp_np.min()) / (pred_disp_np.max() - pred_disp_np.min())

        pred_disp_scaled = (pred_disp.squeeze() * 255).astype(np.uint8)
        
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_depth_map_path = os.path.join(output_dir, "depth_map.jpg")
        depth_image_pil = Image.fromarray(pred_disp_scaled, mode='L') # Create PIL image from numpy array
        depth_image_pil.save(output_depth_map_path, format="JPEG") # Save as JPG

        torch.cuda.empty_cache()
        return Path(output_depth_map_path) # Return cog.Path for output
