# List of augmentations based on randaugment
import random
import PIL.Image
from PIL import Image, ImageFilter, ImageOps, ImageOps, ImageEnhance
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import v2

random_mirror = True


def ShearX(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))


def Identity(img, v):
    return img


def TranslateX(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateXAbs(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYAbs(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return ImageOps.autocontrast(img)


def Invert(img, _):
    return ImageOps.invert(img)


def Equalize(img, _):
    return ImageOps.equalize(img)


def Solarize(img, v):
    return ImageOps.solarize(img, v)


def Posterize(img, v):
    v = int(v)
    return ImageOps.posterize(img, v)


def Contrast(img, v):
    return ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):
    return ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):
    return ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):
    return ImageEnhance.Sharpness(img).enhance(v)


def augment_list():
    l = [
        (Identity, 0, 1),
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Rotate, -30, 30),
        (Solarize, 0, 256),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Brightness, 0.05, 0.95),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.1, 0.1),
        (TranslateX, -0.1, 0.1),
        (TranslateY, -0.1, 0.1),
        (Posterize, 4, 8),
        (ShearY, -0.1, 0.1),
    ]
    return l


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}


class AutoAugment:
    def __init__(self, n):
        self.n = n
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (random.random()) * float(maxval - minval) + minval
            img = op(img, val)

        return img


def get_augment(name):
    return augment_dict[name]


def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        length = random.randint(1, self.length)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Augmentation:
    def __init__(
        self,
        img_size=224,
        val_img_size=256,
        s=1,
        num_aug=4,
        cutout_holes=1,
        cutout_size=75,
        blur=1.0,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ):
        '''self.weak_aug = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    img_size, interpolation=Image.BICUBIC, scale=(0.2, 1.0)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.8 * s, 0.8 * s, 0.4 * s, 0.2 * s)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=blur),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )'''

        self.weak_aug = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.125,0.125)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]
        )

        self.strong_aug = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                AutoAugment(n=num_aug),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                Cutout(n_holes=cutout_holes, length=cutout_size),
            ]
        )
        self.val_aug = transforms.Compose(
            [
                transforms.Resize(
                    (val_img_size, val_img_size), interpolation=Image.BICUBIC
                ),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, x):
        return self.weak_aug(x), self.strong_aug(x), self.val_aug(x)


class Identity_Augmentation:
    def __init__(self, dataset_name='cifar100'):
        if dataset_name == 'cifar10':
            self.mean = [0.4914, 0.4822, 0.4465]
            self.std = [0.2023, 0.1994, 0.2010]
        elif dataset_name == 'cifar100':
            self.mean = [0.5071, 0.4867, 0.4408]
            self.std = [0.2675, 0.2565, 0.2761]
        self.identity_aug = v2.Compose(
            [
                v2.Resize((32, 32), interpolation=Image.BICUBIC, antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self.mean, std=self.std),
            ]
        )
    def __call__(self, x):
        return self.identity_aug(x)


class Weak_Augmentation:
    def __init__(self):
        self.weak_aug = v2.Compose(
            [
                v2.Resize((32, 32), interpolation=Image.BICUBIC, antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomAffine(degrees=0, translate=(0.125,0.125)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ]
        )

    def __call__(self, x):
     return self.weak_aug(x)

class SimCLRaugment:
    def __init__(self, size = (32,32), dataset_name = 'cifar100'):
        self.s = 1

        if dataset_name == 'cifar10':
            self.mean = [0.4914, 0.4822, 0.4465]
            self.std = [0.2023, 0.1994, 0.2010]
        elif dataset_name == 'cifar100':
            self.mean = [0.5071, 0.4867, 0.4408]
            self.std = [0.2675, 0.2565, 0.2761]
        self.size = size
        self.color_jitter = v2.ColorJitter()
        self.data_transforms = v2.Compose([v2.RandomResizedCrop(size=size, antialias=True),
                                           #v2.Resize((32, 32), interpolation=Image.BICUBIC, antialias=True),
                                           v2.RandomHorizontalFlip(),
                                           v2.RandomApply([self.color_jitter], p=0.8),
                                           v2.RandomGrayscale(p=0.2),
                                           v2.ToImage(),
                                           v2.ToDtype(torch.float32, scale=True),
                                           v2.Normalize(mean=self.mean, std=self.std),
                                           ]
                                        )
    
    def __call__(self, x):
        return self.data_transforms(x)











