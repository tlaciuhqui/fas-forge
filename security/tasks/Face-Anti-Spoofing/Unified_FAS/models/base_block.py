"""General FAS building blocks.

Generalized from DCN (``Conv2dBlock``) and ANRL (``Conv_block``/``Basic_block``):

    - ``conv3x3``: standard 3x3 convolution shared by both projects.
    - ``AdaptiveNorm2d``: ANRL's adaptive BN/IN blend, generalized as a
      reusable normalization module.
    - ``ConvBlock``: Conv -> Norm -> Activation with pluggable normalization.
    - ``BasicBlock``: three ``ConvBlock`` layers + 2x max-pool, the small
      downsampling unit shared by the DCN/ANRL encoders.
"""
import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..'))
from common.utils.model_init import init_weights


def conv3x3(in_ch, out_ch, stride=1, padding=1, bias=True):
    """3x3 convolution with symmetric padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=padding, bias=bias)


class AdaptiveNorm2d(nn.Module):
    """Adaptively blend BatchNorm and InstanceNorm features.

    Args:
        num_features (int): number of input channels.
        attention (str|None): '1layer', '2layer', or None for a learnable gate.

    Input:  [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(self, num_features, attention='1layer'):
        super(AdaptiveNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.inorm = nn.InstanceNorm2d(num_features, affine=True)
        self.attention = attention

        if attention == '1layer':
            self.fc = nn.Linear(num_features, num_features)
        elif attention == '2layer':
            self.fc1 = nn.Linear(num_features, num_features // 16 + 1)
            self.fc2 = nn.Linear(num_features // 16 + 1, num_features)
        else:
            self.fc = None
            self.gate = nn.Parameter(torch.full((num_features,), 0.5))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape  # [B, C, H, W]
        x_bn = self.bn(x)
        x_in = self.inorm(x)

        if self.attention is None:
            alpha = self.gate.clamp(0.0, 1.0).view(1, c, 1, 1)
        else:
            z = F.adaptive_avg_pool2d(x, 1).view(b, c)  # [B, C]
            if self.attention == '1layer':
                alpha = self.fc(z)
            else:
                alpha = self.fc2(F.relu(self.fc1(z)))
            alpha = self.sigmoid(alpha).view(b, c, 1, 1)

        return alpha * x_bn + (1.0 - alpha) * x_in  # [B, C, H, W]


class ConvBlock(nn.Module):
    """Conv -> Norm -> Activation block.

    Args:
        in_ch (int): input channels.
        out_ch (int): output channels.
        norm (str): 'bn', 'in' or 'adap'.
        attention (str|None): attention mode when ``norm == 'adap'``.
        activation (str): 'relu' or 'elu'.
        model_init (str): weight initialization method.
    """

    def __init__(self, in_ch, out_ch, norm='bn', attention='1layer',
                 activation='relu', model_init='kaiming'):
        super(ConvBlock, self).__init__()
        self.conv = conv3x3(in_ch, out_ch)

        if norm == 'adap':
            self.norm = AdaptiveNorm2d(out_ch, attention=attention)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        else:
            self.norm = nn.BatchNorm2d(out_ch)

        self.act = nn.ReLU(inplace=True) if activation == 'relu' else nn.ELU(inplace=True)
        init_weights(self.conv, init_type=model_init)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))  # [B, out_ch, H, W]


class BasicBlock(nn.Module):
    """Three ConvBlocks followed by a 2x max-pool (downsample by 2)."""

    def __init__(self, in_ch, out_ch, norm='adap', attention='1layer',
                 activation='relu', model_init='kaiming'):
        super(BasicBlock, self).__init__()
        self.block1 = ConvBlock(in_ch, 128, norm, attention, activation, model_init)
        self.block2 = ConvBlock(128, 196, norm, attention, activation, model_init)
        self.block3 = ConvBlock(196, out_ch, norm, attention, activation, model_init)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.pool(x)  # [B, out_ch, H/2, W/2]
