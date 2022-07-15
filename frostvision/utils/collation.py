import torch


__all__ = ['coco_collate_fn']


def _make_spatial_shape_to_square(tensor_shapes):
    max_shape = tensor_shapes[0]
    for shape in tensor_shapes[1:]:
        for idx, item in enumerate(shape):
            max_shape[idx] = max(max_shape[idx], item)
    return max_shape


def _align_spatial_shape_with_paddings(tensors):
    tensor_shape = _make_spatial_shape_to_square([list(tensor.shape) for tensor in tensors])
    batch_shape = [len(tensors)] + tensor_shape
    padded_tensors = torch.zeros(batch_shape, dtype=tensors[0].dtype, device=tensors[0].device)
    for tensor, padded_tensor in zip(tensors, padded_tensors):
        padded_tensor[: tensor.shape[0], : tensor.shape[1], : tensor.shape[2]].copy_(tensor)
    return padded_tensors


def coco_collate_fn(batch):
    inputs, targets = list(zip(*batch))
    inputs = _align_spatial_shape_with_paddings(inputs)
    return inputs, targets
