from typing import Sequence

import copy
import torch

from torch import Tensor

import torchvision.ops.boxes as B
import torchvision.transforms.functional as T


def crop():
    pass


def flip_horizontally():
    pass


def normalize_tensor_and_target(tensor, target, mean, std, inplace=False):
    tensor = T.normalize(tensor, mean, std, inplace)
    if target is None:
        return tensor, target
    h, w = tensor.shape[-2:]
    if not inplace:
        target = copy.deepcopy(target)
    if 'bboxes' in target:
        bboxes = target['bboxes']
        bboxes = B.box_convert(bboxes, 'xyxy', 'cxcywh')
        bboxes = bboxes / torch.tensor([w, h, w, h], dtype=torch.float32)
        target['bboxes'] = bboxes
    return tensor, target


def pad(tensor, target, padding):
    tensor = T.pad(tensor, [0, 0, padding[0], padding[1]])

    if target is None:
        return tensor, target

    return tensor, ...


def resize_tensor(tensor, size, max_size=None):
    if isinstance(size, Sequence):
        return T.resize(tensor, size)
    if isinstance(size, int):
        h, w = tensor.shape[-2:]
        if max_size:
            long_edge, short_edge = float(max(h, w)), float(min(h, w))
            if long_edge / short_edge * size > max_size:
                size = int(round(max_size * short_edge / long_edge))
        if (h <= w and h == size) or (w <= h and w == size):
            return h, w
        if w < h:
            return int(size * h / w), size
        else:
            return size, int(size * w / h)


def resize_tensor_and_target(tensor, target, size, max_size, inplace=False):
    r"""
    Resize the input tensor to the given size.
    It is expected to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        tensor (Tensor): Tensor to be resized.
        target (dict): Target contains bounding box(es) to be resized.
        size (sequence or int): Desired output size.
        If size is a sequence like (h, w), the output size will be matched to this.
        If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio,
        i.e., if height > width, then image will be rescaled to (size * height / width, size)
        max_size (int, optional): The maximum allowed for the longer edge of the resized tensor.
        If the longer edge of the tensor is greater than ``max_size`` after being resized according to ``size``,
        then the tensor is resized again so that the longer edge is equal to ``max_size``.
        As a result, ``size`` might be overruled, i.e., the smaller edge may be shorter than ``size``.
        This is only supported if ``size`` is an int.
        inplace (bool, optional): Bool to make this operation inplace.

    Returns:
        Tuple[Tensor, dict]: Rescaled tensor and target with rescaled bounding box(es).
    """

    if target is None:
        return tensor, target
    if not inplace:
        target = copy.deepcopy(target)
    if 'bboxes' in target:
        pass
    return ..., ...


def to_tensor(image, target):
    return T.to_tensor(image), target
