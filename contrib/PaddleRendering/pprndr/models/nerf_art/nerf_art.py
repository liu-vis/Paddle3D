#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License")
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Dict, Tuple, Optional

import paddle
import paddle.nn.functional as F
import paddle.nn as nn
import numpy as np
import mcubes

from pprndr.apis import manager
from pprndr.cameras.rays import RayBundle
from pprndr.models.fields import BaseDensityField
from pprndr.models.ray_samplers import BaseSampler
from pprndr.ray_marching import render_alpha_from_densities, render_weights_from_alpha, render_alpha_from_sdf, \
    get_anneal_coeff, get_cosine_coeff
from pprndr.utils.logger import logger

from pprndr.model.criteria.clip_loss import CLIPLoss
from pprndr.model.criteria.patchnce_loss import PatchNCELoss
from pprndr.model.criteria.contrastive_loss import ContrastiveLoss
from pprndr.model.criteria.perp_loss import VGGPerceptualLoss
from pprndr.model.criteria.regularization_loss import RegularizationLoss

__all__ = ["NeRFART"]


@manager.MODELS.add_component
class NeRFART(nn.Layer):
    """
    NeRFART contains two main models: 1) recon_model and 2) style_model.
    """
    def __init__(self,
                 recon_model: nn.Layer,
                 style_model: nn.Layer,
                 image_width: int,
                 image_height: int,
                 window_size: int = 256,
                 loss_weight_eikonal: float = 0.1,
                 loss_weight_clip: float = 1.0,
                 loss_weight_contrastive: float = 0.2,
                 loss_weight_patchnce: float = 0.1,
                 loss_weight_perceptual: float = 0.2,

                 loss_weight_idr: float = 1.0,
                 loss_weight_mask: float = 1.0,
                 anneal_end: float = 0.0,
                 target_hw: list = None, 
                 neg_prompt_file: str = "pprndr/models/criteria/neg_text.txt"
                 prompt_for_recon: str = "",
                 prompt_for_style: str = ""
    ):
        self.recon_model = recon_model
        self.recon_model.eval() 
        self.style_model = style_model

        self.im_width = image_width
        self.im_height = image_height
        self.window_size = window_size
        self.style_image_grads = None

        # Criterias
        if target_hw is None:
            target_hw = [960, 540]  # full-res 4:3
        self.neg_texts = neg_prompt_file
        self.recon_prompt = prompt_for_recon
        self.style_prompt = prompt_for_style
        self.clip_loss = CLIPLoss()
        self.perp_loss = VGGPerceptualLoss()
        self.reg_loss = RegularizationLoss()
        self.contrastive_loss = ContrastiveLoss(neg_text=self.neg_texts)
        self.patchnce_loss = PatchNCELoss(target_hw, self.neg_texts)
        
        # Loss weights
        self.loss_weight_eikonal = loss_weight_eikonal
        self.loss_weight_clip = loss_weight_clip
        self.loss_weight_contrastive = loss_weight_contrastive
        self.loss_weight_patchnce = loss_weight_patchnce
        self.loss_weight_perceptual = loss_weight_perceptual
        
        
        super(NeRFART, self).__init__()

    def compute_style_image_grads(self, ray_bundle):
        # Get the number of rays
        num_rays = len(ray_bundle) # (= the # of pixels of an entire image)

        # Generate recon_image and style_image
        with paddle.no_grad():
            # Step1: Inference by the reconstruction model in eval mode
            temp_recon_outputs = defaultdict(list)
            for b_id in range(0, num_rays, self.window_size):
                cur_ray_bundle = ray_bundle[b_id:b_id + window_size]
                outputs = self.recon_model(cur_ray_bundle)
                for k, v in outputs.items():
                    temp_recon_outputs[k].append(v)

            recon_outputs = {} # image, gradients
            for k, v in temp_recon_outputs.items():
                if isinstance(v[0], paddle.Tensor):
                    recon_outputs[k] = paddle.concat(v, axis=0)
                else:
                    recon_outputs[k] = v
            recon_image = recon_outputs['rgb'].reshape([-1, self.im_height, self.im_width, 3])
            recon_image = recon_image.transpose([0, 3, 1, 2])
            
            # Step2: Inference by the stylization model in eval mode
            self.style_model.eval()
            temp_style_outputs = defaultdict(list)
            for b_id in range(0, num_rays, self.window_size):
                cur_ray_bundle = ray_bundle[b_id:b_id + window_size]
                outputs = self.style_model(cur_ray_bundle) 
                for k, v in outputs.items():
                    temp_style_outputs[k].append(v)
                    
            style_outputs = {} # image, gradients
            for k, v in temp_style_outputs.items():
                if isinstance(v[0], paddle.Tensor):
                    style_outputs[k] = paddle.concat(v, axis=0)
                else:
                    style_output[k] = v
            style_image = style_outputs['rgb'].reshape([-1, self.im_height, self.im_width, 3])
            style_image = style_image.transpose([0, 3, 1, 2])
            style_image = style_image.detach() # Not sure if this is necessary.

        self.style_image.stop_gradient = False

        # Compute image level losses
        clip_loss = self.clip_loss(recon_image, self.recon_prompt, style_image, self.style_prompt)
        clip_loss = self.loss_weight_clip * clip_loss

        contra_loss = self.contrastive_loss(style_image, self.style_prompt)
        contra_loss = self.loss_weight_contrastive * contra_loss

        patchnce_loss = self.patchnce_loss(style_image, self.style_prompt)
        patchnce_loss = self.loss_weight_patchnce * patchnce_loss

        perp_loss = self.perceptual_loss(recon_image, style_image)
        perp_loss = self.loss_weight_perceptual * perp_loss

        # We omitted the regularization loss here because we have implemented the mask_loss in the NeuS model, which provides a similar constraint to training. [To confirm its correctness]
        # loss_regular = Cancel.
        
        imlevel_loss = clip_loss + contra_loss + perp_loss
        imlevel_loss.backward() # backward upto style_image
        return style_image.grad.clone().reshape([-1, self.num_rays, 3])
        

    def _forward_art(self, 
                     sample, 
                     sub_sample = None, 
                     cur_iter = None, 
                     cur_sub_iter = None):

        if cur_sub_iter is not None: # means "iter_size" was enabled.
            assert(self.training)
            assert(sub_sample is not None)
            input_sample = sub_sample
            if cur_sub_iter == 0:
                entire_ray_bundle, entire_pixel_batch = sample # For the entire image

                # Compute image level grads. at sub_iter = 0. 
                self.style_image_grads = self.compute_style_image_grads(entire_ray_bundle)
                # [To comfirm: do we need clean something(e.g., del style_image) here?]

        else: # means "iter_size" was disabled.
            assert(sub_sample is None)
            assert(cur_iter is None)
            assert(not self.training)
            input_sample = sample
        
        outputs = self.style_model(input_sample, cur_iter)
        if self.training:
            num_rays = len(input_sample[0])

            # Format: outputs['grad_back'][NAME] = [TensorA, TensorA's grad]
            outputs['grad_back']["rgb_grad"] = [outputs["rgb"], 
                                                self.style_image_grad[:, cur_sub_iter:cur_sub_iter + num_rays, :]]
        return outputs


    def forward(self, *args, **kwargs):
        if hasattr(self, "amp_cfg_") and self.training:
            with paddle.amp.auto_cast(**self.amp_cfg_):
                return self._forward_art(*args, **kwargs)        
        return self._forward_art(*args, **kwargs)
