"""
pytorch implementation of the commonlayer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

weight_regularizer = nn.L1Loss()  # Adjust as needed for L2 regularization


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    return nn.Parameter(torch.empty(*shape).uniform_(-scale, scale))


# Xavier initialize
def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    return nn.Parameter(nn.init.xavier_uniform_(torch.empty(*shape)))


def zeros(shape, name=None):
    """All zeros."""
    return nn.Parameter(torch.zeros(*shape))


def ones(shape, name=None):
    """All ones."""
    return nn.Parameter(torch.ones(*shape))


class DotProduct(nn.Module):
    def __init__(self, sparse=False):
        super().__init__()
        self.sparse = sparse

    def forward(self, x, y):
        if self.sparse:
            if not x.is_sparse:
                x = x.to_sparse()
            return torch.sparse.mm(x, y)
        else:
            return torch.mm(x, y)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, is_training=True, kx=3, ky=3,
                 stride_x=2, stride_y=2, batchnorm=False, padding='VALID', add=None, deconv=False, activation='relu'):
        super().__init__()
        self.is_training = is_training
        self.add = add
        self.batchnorm = batchnorm

        # PyTorch中对VALID和SAME的填充进行适配
        if padding == 'VALID':
            self.padding = (0, 0)
        elif padding == 'SAME':
            self.padding = "same"
        elif padding == '1':
            self.padding = (1, 1)
        else:
            raise ValueError(f"Unsupported padding mode: {padding}")

        if not deconv:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kx, ky), stride=(stride_x, stride_y),
                                  padding=self.padding, bias=not batchnorm)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(kx, ky), stride=(stride_x, stride_y),
                                           padding=self.padding, bias=not batchnorm)

        self.bn_layer = nn.BatchNorm2d(out_channels) if batchnorm else None

        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation_fn = nn.Sigmoid()
        else:
            self.activation_fn = None

    def forward(self, x):
        if self.add is not None:
            if not isinstance(self.add, torch.Tensor):
                raise ValueError("'add' must be a tensor.")
            if self.add.shape != x.shape:
                raise ValueError("The shape of 'add' must match the shape of input 'x'.")
            x = x + self.add

        x = self.conv(x)

        if self.batchnorm:
            x = self.bn_layer(x)

        if self.activation_fn is not None:
            x = self.activation_fn(x)

        return x
