import frostvision.transforms.coco as transforms
import frostvision.utils as utils

from torch.utils.data import DataLoader
from frostvision.datasets import COCODetection


def main():
    root = '/mnt/datasets/coco/train2017'
    annFile = '/mnt/datasets/coco/annotations/instances_train2017.json'
    transforms_coco = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(640, max_size=1333),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    trainset = COCODetection(root, annFile, transforms_coco)
    trainloader = DataLoader(trainset, 4, True, num_workers=0, collate_fn=utils.collation.coco_collate_fn)

    denorm = transforms.Broadcast([transforms.Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    inputs, targets = next(iter(trainloader))
    inputs, targets = denorm((inputs, targets))
    utils.visualize.visualize_bounding_boxes_on_batch((inputs, targets))


if __name__ == '__main__':
    main()
