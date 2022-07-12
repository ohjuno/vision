from typing import Callable, Optional

import os.path

from torchvision.datasets import ImageFolder


__all__ = ['ImageNet']


class ImageNet(ImageFolder):

    def __init__(self, root: str, train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        folder = 'train' if train else 'val'
        root = os.path.join(root, folder)
        super(ImageNet, self).__init__(root, transform, target_transform)
