"""Dataset factory for the general CelebA-Spoof FAS project."""
import os
import sys

import albumentations as alb
from omegaconf import OmegaConf

from .dataset import CelebASpoofDataset

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..'))
from common.data import create_base_transforms, create_base_dataloader


def create_data_transforms(args, split='train'):
    """Create transforms for a given split.

    The light HSV jitter is a general FAS augmentation shared by DCN/ANRL.
    """
    base_transform = create_base_transforms(args, split=split)
    if split == 'train':
        aug_transform = alb.Compose([
            alb.HueSaturationValue(p=0.1),
        ])
        return alb.Compose([*aug_transform, *base_transform])
    return base_transform


def create_dataloader(args, split='train', category=None, print_info=False, **overrides):
    """Create a dataloader for the configured CelebASpoofDataset.

    ``overrides`` can replace dataset config keys at runtime, e.g. to build a
    loader with a different list path or split.
    """
    dataset_cfg = getattr(args.dataset, args.dataset.name)
    kwargs = OmegaConf.to_container(dataset_cfg, resolve=True)
    kwargs.update(overrides)

    transform = create_data_transforms(args.transform, split=split)

    dataset = CelebASpoofDataset(transform=transform,
                                 split=split,
                                 category=category,
                                 print_info=print_info,
                                 **kwargs)
    dataloader = create_base_dataloader(args, dataset, split=split)
    return dataloader
