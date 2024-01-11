from typing import Any
from torchvision.transforms import v2
from PIL import Image
import torch
from lightly.transforms.utils import IMAGENET_NORMALIZE
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


class Identity_Augmentation:
    def __init__(self):
        self.identity_aug = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std']),
            ]
        )
    def __call__(self, x):
        return self.identity_aug(x)
    

class SimCLRaugment:
    def __init__(self, size = (64,64)):
        self.s = 1
        self.size = size
        self.color_jitter = v2.ColorJitter()
        self.data_transforms = v2.Compose([
                                           v2.RandomHorizontalFlip(),
                                           v2.RandomApply([self.color_jitter], p=0.8),
                                           v2.RandomGrayscale(p=0.2),
                                           v2.ToImage(),
                                           v2.ToDtype(torch.float32, scale=True),
                                           v2.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std']),
                                           ]
                                        )
    
    def __call__(self, x):
        return self.data_transforms(x)


class Weak_Augmentation:
    def __init__(self):
        self.weak_aug = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomAffine(degrees=0, translate=(0.125,0.125)),
                # v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std']),
            ]
        )

    def __call__(self, x):
     return self.weak_aug(x)



class RandAugment:
    def __init__(self, num_ops: int=4):
        self.num_ops = num_ops
        self.t = v2.Compose(
            [
                v2.RandAugment(num_ops=self.num_ops, interpolation=transforms.InterpolationMode.BILINEAR),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_NORMALIZE['mean'], std=IMAGENET_NORMALIZE['std']),
            ]
        )

    def __call__(self, x):
        return self.t(x)


