from typing import Any, List, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

from torch import Tensor
from torchvision.utils import draw_bounding_boxes

plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0
plt.rcParams['figure.figsize'] = (50, 50)


def show(images: Union[Tensor, np.ndarray, List[Union[Tensor, np.ndarray]]]) -> None:
    if not isinstance(images, list):
        images = [images]
    fix, axs = plt.subplots(nrows=len(images)//4, ncols=4, squeeze=False)
    for idx, image in enumerate(images):
        image = F.to_pil_image(image)
        axs[idx//4, idx%4].imshow(np.asarray(image))
        axs[idx//4, idx%4].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def cast_float32_to_uint8(t: Tensor) -> Tensor:
    return (t * 255).type(torch.uint8)


def visualize_bounding_boxes_on_batch(batch: Tuple[Tensor, Any]) -> None:
    images, targets = batch
    images = images.detach()
    images = cast_float32_to_uint8(images)
    results = []
    for image, target in zip(images, targets):
        results.append(draw_bounding_boxes(image, target['bboxes']))
    show(results)
