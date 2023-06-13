#  !/usr/bin/env python3
"""
infonce loss functions
"""

from paddlenlp.transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
import paddle
import paddle.vision.transforms as transforms
import paddle.nn.functional as F
import paddle.nn as nn


class InforNCELoss(nn.Layer):
    """
    InforNCELoss Class
    """
    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super(InforNCELoss, self).__init__()

    def forward(self, query, positive_key, negative_keys=None):
        """
        Args:
            query (str): 
        """
        pass