from torchvision.transforms import v2
from PIL import Image
import torch
from lightly.transforms.utils import IMAGENET_NORMALIZE


class Identity_Augmentation:
    def __init__(self):
        self.identity_aug = v2.Compose(
            [
                v2.Resize((32, 32), interpolation=Image.BICUBIC, antialias=True),
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
        self.data_transforms = v2.Compose([v2.RandomResizedCrop(size=size, antialias=True),
                                           #v2.Resize((32, 32), interpolation=Image.BICUBIC, antialias=True),
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


