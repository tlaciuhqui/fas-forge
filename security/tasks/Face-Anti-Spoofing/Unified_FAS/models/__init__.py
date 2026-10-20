"""Model components for the general CelebA-Spoof FAS project."""
from .framework import (
    UnifiedFAS,
    FeatureExtractor,
    Classifier,
    DepthEstimator,
    DINOv2RegFeatureExtractor,
)
from .base_block import ConvBlock, BasicBlock, AdaptiveNorm2d, conv3x3
