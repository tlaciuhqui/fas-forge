"""General Face Anti-Spoofing model.

``UnifiedFAS`` is a compact encoder + classifier + auxiliary-map architecture
that captures the *general* design shared by DCN and ANRL:

    - A small CNN feature extractor producing multi-scale features (both DCN
      and ANRL use a ~3-stage, 384-channel encoder).
    - A binary classifier on pooled features (ANRL's FeatEmbedder).
    - An optional depth/reflection map head for auxiliary supervision (DCN's
      Auxilary_Deep / ANRL's DepthEstmator).

Method-specific components (DCN's patch-similarity head and ANRL's meta-learned
adaptive-norm update) are not re-implemented here; they live in their own
projects. The adaptive normalization block itself is kept as an optional
encoder normalization, since it is a reusable general building block.
"""
import os
import sys

import timm
import torch
from torch import nn
import torch.nn.functional as F

from .base_block import ConvBlock, BasicBlock

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..'))
from common.utils.model_init import init_weights


class FeatureExtractor(nn.Module):
    """Small multi-scale CNN encoder.

    Input:  [B, in_ch, H, W]
    Output: (cat_feat [B, mid_ch, H/8, W/8], deep_feat [B, mid_ch/3, H/8, W/8])
    """

    def __init__(self, in_ch=6, mid_ch=384, norm='adap', attention='1layer',
                 model_init='kaiming'):
        super(FeatureExtractor, self).__init__()
        assert mid_ch % 3 == 0, 'mid_ch must be divisible by 3'
        out_ch = mid_ch // 3

        self.inc = ConvBlock(in_ch, 64, norm=norm, attention=attention, model_init=model_init)
        self.down1 = BasicBlock(64, out_ch, norm, attention, model_init=model_init)
        self.down2 = BasicBlock(out_ch, out_ch, norm, attention, model_init=model_init)
        self.down3 = BasicBlock(out_ch, out_ch, norm, attention, model_init=model_init)

    def forward(self, x):
        dx1 = self.inc(x)       # [B, 64, H, W]
        dx2 = self.down1(dx1)   # [B, C, H/2, W/2]
        dx3 = self.down2(dx2)   # [B, C, H/4, W/4]
        dx4 = self.down3(dx3)   # [B, C, H/8, W/8]

        target_size = dx4.shape[-2:]
        re_dx2 = F.adaptive_avg_pool2d(dx2, target_size)  # [B, C, H/8, W/8]
        re_dx3 = F.adaptive_avg_pool2d(dx3, target_size)  # [B, C, H/8, W/8]
        cat_feat = torch.cat([re_dx2, re_dx3, dx4], dim=1)  # [B, 3C, H/8, W/8]
        return cat_feat, dx4


class Classifier(nn.Module):
    """Binary classifier head.

    Input:  [B, in_ch, H, W]
    Output: [B, num_classes]
    """

    def __init__(self, in_ch=384, num_classes=2, norm='bn', model_init='kaiming'):
        super(Classifier, self).__init__()
        self.block1 = ConvBlock(in_ch, 128, norm=norm, model_init=model_init)
        self.block2 = ConvBlock(128, 256, norm=norm, model_init=model_init)
        self.block3 = ConvBlock(256, 512, norm=norm, model_init=model_init)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        init_weights(self.fc, init_type=model_init)

    def forward(self, x):
        x = self.block1(x)
        x = F.max_pool2d(x, 2)
        x = self.block2(x)
        x = F.max_pool2d(x, 2)
        x = self.block3(x)
        x = self.pool(x).flatten(1)  # [B, 512]
        return self.fc(x)            # [B, num_classes]


class DepthEstimator(nn.Module):
    """Auxiliary depth/reflection map head.

    Input:  [B, in_ch, H, W]
    Output: [B, out_ch, H, W]
    """

    def __init__(self, in_ch=384, out_ch=1, norm='bn', model_init='kaiming'):
        super(DepthEstimator, self).__init__()
        self.block1 = ConvBlock(in_ch, 128, norm=norm, model_init=model_init)
        self.block2 = ConvBlock(128, 64, norm=norm, model_init=model_init)
        self.block3 = ConvBlock(64, out_ch, norm=norm, model_init=model_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.block3(x)  # [B, out_ch, H, W]


class DINOv2RegFeatureExtractor(nn.Module):
    """DINOv2 feature extractor with register tokens, via ``timm``.

    timm exposes the register-token variants of DINOv2 as ``*_reg4_*`` models,
    e.g. ``vit_small_patch14_reg4_dinov2``. In those models:

        - ``backbone.cls_token`` is the [CLS] token (shape [1, 1, embed_dim]).
        - ``backbone.reg_token`` holds the learnable register tokens
          (shape [1, num_registers, embed_dim]).
        - ``backbone.num_prefix_tokens`` = 1 + num_registers.

    ``forward_features`` therefore returns a sequence of
    ``[B, 1 + num_registers + num_patches, embed_dim]`` tokens. For FAS we drop
    both [CLS] and the register tokens and keep only patch tokens, reshaping
    them into a 2-D feature map that the CNN classifier/aux heads can consume.

    Args:
        model_name (str): timm DINOv2-with-registers model name.
        img_size (int): input size. Must be divisible by the patch size (14).
        pretrained (bool): load pretrained DINOv2 weights from the timm hub.
        freeze (bool): freeze the backbone parameters.

    Input:  [B, 3, H, W]  (H/W divisible by 14)
    Output: (feat [B, embed_dim, H/14, W/14], feat)
    """

    def __init__(self, model_name='vit_small_patch14_reg4_dinov2', img_size=224,
                 pretrained=False, freeze=False):
        super(DINOv2RegFeatureExtractor, self).__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # drop the classification head
            img_size=img_size,
            dynamic_img_size=True,  # accept 14-divisible input sizes
        )
        self.embed_dim = self.backbone.embed_dim
        self.num_prefix_tokens = self.backbone.num_prefix_tokens

        reg_token = getattr(self.backbone, 'reg_token', None)
        self.num_registers = reg_token.shape[1] if reg_token is not None else 0

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        tokens = self.backbone.forward_features(x)  # [B, 1 + R + N, C]
        # Drop [CLS] (position 0) and the R register tokens; keep patch tokens.
        patch_tokens = tokens[:, self.num_prefix_tokens:]  # [B, N, C]

        b, n, c = patch_tokens.shape
        h = w = int(n ** 0.5)
        assert h * w == n, f'Expected a square patch grid, got {n} patches'

        feat = patch_tokens.transpose(1, 2).reshape(b, c, h, w)  # [B, C, H, W]
        return feat, feat


class UnifiedFAS(nn.Module):
    """General FAS model: encoder + classifier + optional aux head.

    Args:
        in_ch (int): input channels (3 for rgb, 6 for rgb_hsv/rgb_ycrcb).
        mid_ch (int): CNN encoder output channels (ignored for DINOv2).
        num_classes (int): number of classes.
        norm (str): encoder normalization ('bn', 'in', 'adap').
        attention (str|None): adaptive-norm attention mode.
        aux_head (bool): enable the auxiliary depth/reflection head.
        aux_out_ch (int): auxiliary head output channels.
        model_init (str): weight initialization method.
        backbone (str): 'cnn' or 'dinov2_reg'.
        backbone_model (str): timm model name when ``backbone == 'dinov2_reg'``.
        backbone_pretrained (bool): load pretrained DINOv2 weights.
        backbone_freeze (bool): freeze the DINOv2 backbone.
        backbone_img_size (int): input size for the DINOv2 backbone.

    Input:  [B, in_ch, H, W]
    Output: (logits [B, num_classes], aux [B, aux_out_ch, H', W']|None, feat)
    """

    def __init__(self, in_ch=6, mid_ch=384, num_classes=2, norm='adap',
                 attention='1layer', aux_head=True, aux_out_ch=1,
                 model_init='kaiming', backbone='cnn',
                 backbone_model='vit_small_patch14_reg4_dinov2',
                 backbone_pretrained=False, backbone_freeze=False,
                 backbone_img_size=224):
        super(UnifiedFAS, self).__init__()

        if backbone == 'cnn':
            self.features = FeatureExtractor(in_ch=in_ch, mid_ch=mid_ch, norm=norm,
                                             attention=attention, model_init=model_init)
            feat_ch = mid_ch
        elif backbone == 'dinov2_reg':
            self.features = DINOv2RegFeatureExtractor(
                model_name=backbone_model,
                img_size=backbone_img_size,
                pretrained=backbone_pretrained,
                freeze=backbone_freeze,
            )
            feat_ch = self.features.embed_dim
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

        self.backbone = backbone
        self.feat_ch = feat_ch
        self.classifier = Classifier(in_ch=feat_ch, num_classes=num_classes,
                                     norm='bn', model_init=model_init)
        self.aux_head = aux_head
        if aux_head:
            self.depth_estimator = DepthEstimator(in_ch=feat_ch, out_ch=aux_out_ch,
                                                  norm='bn', model_init=model_init)

    def forward(self, x):
        feat, _ = self.features(x)                     # [B, feat_ch, H', W']
        logits = self.classifier(feat)                 # [B, num_classes]
        aux = self.depth_estimator(feat) if self.aux_head else None  # [B, 1, H', W']
        return logits, aux, feat
