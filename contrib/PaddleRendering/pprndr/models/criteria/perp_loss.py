#  !/usr/bin/env python3
"""
VGGPerceptualLoss functions
"""
import paddle
import paddle.nn as nn
from paddle.vision.models import vgg16
from utils.download import get_path_from_url

import numpy as np

__all__ = ['VGGPerceptualLoss']

VGG16_TORCHVISION_URL = 'https://paddlegan.bj.bcebos.com/models/vgg16_official.pdparams'


class VGGPerceptualLoss(nn.Layer):
    """
    VGGPerceptualLoss functions
    """
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg16_model = vgg16(pretrained=False)
        weight_path = get_path_from_url(VGG16_TORCHVISION_URL)
        state_dicts = paddle.load(weight_path)
        
        vgg16_model.set_state_dict(state_dicts)
        vgg16_model.eval()
        
        blocks = []
        blocks.append(vgg16_model.features[:4])
        blocks.append(vgg16_model.features[4:9])
        blocks.append(vgg16_model.features[9:16])
        blocks.append(vgg16_model.features[16:23])
        for bl in blocks:
            for p in bl.parameters():
                p.stop_gradient = True
        self.blocks = paddle.nn.LayerList(blocks)
        self.transform = paddle.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", paddle.to_tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]))
        self.register_buffer("std", paddle.to_tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]))

    def forward(self, I_src, I_tgt, feature_layers=[3], style_layers=[]):
        
        I_src = (I_src-self.mean) / self.std
        I_tgt = (I_tgt-self.mean) / self.std
        if self.resize:
            I_src = self.transform(I_src, mode='bilinear', size=(224, 224), align_corners=False)
            I_tgt = self.transform(I_tgt, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = I_src
        y = I_tgt
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += paddle.nn.functional.mse_loss(x, y)
            if i in style_layers:
                act_x = x.reshape([x.shape[0], x.shape[1], -1])
                act_y = y.reshape([y.shape[0], y.shape[1], -1])
                gram_x = act_x @ act_x.transpose([0, 2, 1])
                gram_y = act_y @ act_y.transpose([0, 2, 1])
                loss += paddle.nn.functional.l1_loss(gram_x, gram_y)
        return loss


if __name__ == "__main__":
    percep = VGGPerceptualLoss()

    a = paddle.ones([1, 3, 300, 300])
    b = paddle.zeros([1, 3, 300, 300])
    loss = percep(a, b)
    print(loss)