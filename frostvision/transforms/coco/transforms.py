import random
from typing import Sequence

import torch
from torch.nn import Module

import torchvision.transforms as T
import frostvision.transforms.coco.functional as F


class Broadcast:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, batch):
        i, j = [], []
        inputs, targets = batch
        for tensor, target in zip(inputs, targets):
            for t in self.transforms:
                tensor, target = t(tensor, target)
            i.append(tensor); j.append(target)
        return torch.stack(i), tuple(j)


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class CenterCrop(Module):

    def __init__(self, size, inplace=False):
        super(CenterCrop, self).__init__()
        self.size = size if isinstance(size, (list, tuple)) else (size, size)
        self.inplace = inplace

    def forward(self, tensor, target):
        th, tw = tensor.shape[-2:]
        ch, cw = self.size
        i = int(round((th - ch) / 2.))
        j = int(round((tw - cw) / 2.))
        return F.crop(tensor, target, (i, j, ch, cw), self.inplace)


class Denormalize(Module):

    def __init__(self, mean, std, inplace=False):
        super(Denormalize, self).__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor, target):
        return F.denormalize_tensor_and_target(tensor, target, self.mean, self.std, self.inplace)


class Normalize(Module):

    def __init__(self, mean, std, inplace=False):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor, target):
        return F.normalize_tensor_and_target(tensor, target, self.mean, self.std, self.inplace)


class RandomCrop(Module):

    def __init__(self, size, inplace=False):
        super(RandomCrop, self).__init__()
        self.size = size
        self.inplace = inplace

    def forward(self, tensor, target):
        region = T.RandomCrop.get_params(tensor, self.size)
        return F.crop(tensor, target, region, self.inplace)


class RandomErasing(Module):

    def __init__(self, *args, **kwargs):
        super(RandomErasing, self).__init__()
        self.eraser = T.RandomErasing(*args, **kwargs)

    def forward(self, tensor, target):
        return self.eraser(tensor), target


class RandomHorizontalFlip(Module):

    def __init__(self, p=0.5, inplace=False):
        super(RandomHorizontalFlip, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, tensor, target):
        if random.random() < self.p:
            return F.flip_horizontally(tensor, target, self.inplace)
        return tensor, target


class RandomPad(Module):

    def __init__(self, max_pad, inplace=False):
        super(RandomPad, self).__init__()
        self.max_pad = max_pad
        self.inplace = inplace

    def forward(self, tensor, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return F.pad(tensor, target, (pad_x, pad_y), self.inplace)


class RandonResize(Module):

    def __init__(self, sizes, max_size=None, inplace=False):
        super(RandonResize, self).__init__()
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size
        self.inplace = inplace

    def forward(self, tensor, target):
        size = random.choice(self.sizes)
        return F.resize_tensor_and_target(tensor, target, size, self.max_size, self.inplace)


class RandomResizedCrop(Module):
    pass


class RandomSelect(Module):

    def __init__(self, transforms_1, transforms_2, p=0.5):
        super(RandomSelect, self).__init__()
        self.transforms_1 = transforms_1
        self.transforms_2 = transforms_2
        self.p = p

    def forward(self, tensor, target):
        if random.random() < self.p:
            return self.transforms_1(tensor, target)
        return self.transforms_2(tensor, target)


class Resize(Module):

    def __init__(self, size, max_size):
        super(Resize, self).__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError
        self.size = size
        self.max_size = max_size

    def forward(self, tensor, target):
        return F.resize_tensor_and_target(tensor, target, self.size, self.max_size)


class ToTensor:

    def __call__(self, image, target):
        return F.to_tensor(image, target)
