import random
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import v2

class Identity_Augmentation:
    def __init__(self):
        self.identity_aug = v2.Compose(
            [
                v2.Resize((32, 32), interpolation=Image.BICUBIC, antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True)
            ]
        )
    def __call__(self, x):
        return self.identity_aug(x)


class Weak_Augmentation:
    def __init__(self):
        self.weak_aug = v2.Compose(
            [
                v2.Resize((32, 32), interpolation=Image.BICUBIC),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomAffine(degrees=0, translate=(0.125,0.125)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True)
            ]
        )

    def __call__(self, x):
     return self.weak_aug(x)



class SimCLRaugment:
    def __init__(self, size = (32,32)):
        self.s = 1
        self.size = size
        self.color_jitter = v2.ColorJitter()
        self.data_transforms = v2.Compose([v2.RandomResizedCrop(size=size, antialias=True),
                                           #v2.Resize((32, 32), interpolation=Image.BICUBIC, antialias=True),
                                           v2.RandomHorizontalFlip(),
                                           v2.RandomApply([self.color_jitter], p=0.8),
                                           v2.RandomGrayscale(p=0.2),
                                           v2.ToImage(),
                                           v2.ToDtype(torch.float32, scale=True)])
    
    def __call__(self, x):
        return self.data_transforms(x)







