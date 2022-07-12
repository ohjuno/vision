from typing import Sequence

from torch.nn import Module

import frostvision.transforms.coco.functional as F


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:

    def __call__(self, image, target):
        return F.to_tensor(image, target)


class Normalize(Module):

    def __init__(self, mean, std, inplace=False):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor, target):
        return F.normalize_tensor_and_target(tensor, target, self.mean, self.std, self.inplace)


class Resize(Module):

    def __init__(self, size, max_size):
        super(Resize, self).__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError
        self.size = size
        self.max_size = max_size

    def forward(self, image, target):
        return F.resize_tensor_and_target(image, target, self.size, self.max_size)
