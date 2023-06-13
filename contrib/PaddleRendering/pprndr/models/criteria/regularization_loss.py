#  !/usr/bin/env python3
"""
Regulariation methods
"""
import paddle
from PIL import Image
import paddle.nn.functional as F
import paddle.nn as nn


class RegularizationLoss(nn.Layer):
    """
    Regulariation Class
    """

    def __init__(self):
        super(RegulariationLoss, self).__init__()

    def forward(self, weight, samples):
        '''
        weight: [B, N]
        samples:[B, N+1]
        '''
        S = 16
        B = weight.shape[0]//S
        loss_bi = 0

        for i in range(S):
            w = weight[i * B:(i + 1) * B]
            s = samples[i * B:(i + 1) * B]
            ww = w.unsqueeze(-1) * w.unsqueeze(-2)          # [B,N,N]
            mm = (s.unsqueeze(-1) - s.unsqueeze(-2)).abs()  # [B,N,N]
            loss = (ww * mm).sum((-1, -2)).mean()
            loss_bi += loss
        loss_bi /= S
        return loss_bi


if __name__ == "__main__":
    a = paddle.ones([1280, 128])
    b = paddle.ones([1280, 128]) * 1.5

    reg_loss = RegulariationLoss()

    loss = reg_loss(a, b)
    print("loss:  ", loss)
