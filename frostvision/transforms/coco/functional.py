from typing import Sequence

import copy
import torch

from torch import Tensor

import torchvision.ops.boxes as B
import torchvision.transforms.functional as T


def crop(tensor, target, region, inplace=False):
    tensor = T.crop(tensor, *region)
    if target is None:
        return tensor, target
    if not inplace:
        target = copy.deepcopy(target)
    i, j, h, w = region
    target['size'] = torch.tensor([h, w])
    fields = ['labels', 'area', 'iscrowd']
    if 'bboxes' in target:
        bboxes = target['bboxes']
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_bboxes = bboxes - torch.as_tensor([j, i, j, i])
        cropped_bboxes = torch.min(cropped_bboxes.reshape(-1, 2, 2), max_size)
        cropped_bboxes = cropped_bboxes.clamp(min=0)
        target['bboxes'] = cropped_bboxes.reshape(-1, 4)
        fields.append('bboxes')
        if 'area' in target:
            area = (cropped_bboxes[:, 1, :] - cropped_bboxes[:, 0, :]).prod(dim=1)
            target['area'] = area
    if 'mask' in target:
        pass
    if 'bboxes' in target or 'mask' in target:
        if 'bboxes' in target:
            cropped_bboxes = target['bboxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_bboxes[:, 1, :] > cropped_bboxes[:, 0, :], dim=1)
        else:
            keep = ...
        for field in fields:
            target[field] = target[field][keep]
    return tensor, target


@torch.no_grad()
def denormalize_tensor_and_target(tensor, target, mean, std, inplace=False):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    if target is None:
        return tensor, target
    h, w = target['size']
    if not inplace:
        target = copy.deepcopy(target)
    if 'bboxes' in target:
        bboxes = target['bboxes']
        bboxes = bboxes * torch.as_tensor([w, h, w, h], dtype=torch.float32)
        bboxes = B.box_convert(bboxes, 'cxcywh', 'xyxy')
        target['bboxes'] = bboxes
    return tensor, target


def flip_horizontally(tensor, target, inplace=False):
    tensor = T.hflip(tensor)
    h, w = tensor.shape[-2:]
    if target is None:
        return tensor, target
    if not inplace:
        target = copy.deepcopy(target)
    if 'bboxes' in target:
        bboxes = target['bboxes']
        bboxes = bboxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target['bboxes'] = bboxes
    if 'mask' in target:
        pass
    return tensor, target


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


def pad(tensor, target, padding, inplace=False):
    tensor = T.pad(tensor, [0, 0, padding[0], padding[1]])
    if target is None:
        return tensor, target
    if not inplace:
        target = copy.deepcopy(target)
    target['size'] = torch.tensor(tensor.shape[-2:])
    if 'mask' in target:
        pass
    return tensor, target


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
            return T.resize(tensor, (h, w))
        if w < h:
            return T.resize(tensor, (int(size * h / w), size))
        else:
            return T.resize(tensor, (size, int(size * w / h)))


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
    rescaled_tensor = resize_tensor(tensor, size, max_size)
    if target is None:
        return rescaled_tensor, target
    scale = (float(r) / float(t) for r, t in zip(rescaled_tensor.shape[-2:], tensor.shape[-2:]))
    scale_x, scale_y = scale
    if not inplace:
        target = copy.deepcopy(target)
    if 'bboxes' in target:
        bboxes = target['bboxes']
        bboxes = bboxes * torch.as_tensor([scale_x, scale_y, scale_x, scale_y])
        target['bboxes'] = bboxes
    if 'area' in target:
        area = target['area']
        area = area * (scale_x * scale_y)
        target['area'] = area
    h, w = rescaled_tensor.shape[-2:]
    target['size'] = torch.tensor([h, w])
    if 'mask' in target:
        pass  # for segmentation task, future work
    return rescaled_tensor, target


def to_tensor(image, target):
    return T.to_tensor(image), target
