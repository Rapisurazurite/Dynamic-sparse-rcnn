import argparse
from random import random
from typing import Sequence, Dict, Tuple, Union

import cv2
import numpy as np
import torch
from albumentations import DualTransform
from matplotlib import pyplot as plt

from sparse_rcnn.utils.config import cfg_from_yaml_file, cfg
from sparse_rcnn.dataloader.dataset import CocoDataset
import albumentations as A
from albumentations.augmentations.geometric import functional as F

parser = argparse.ArgumentParser(description="Test coco dataset")
coco_config = "../sparse_rcnn/configs/coco.yaml"
cfg_from_yaml_file(coco_config, cfg)


transforms = A.Compose([
    A.SmallestMaxSize(max_size=cfg.INPUT.MIN_SIZE_TRAIN, p=1),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=30, p=0.5),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.2)],
    # x_min, y_min, x_max, y_max
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=["classes"])
)


dataset = CocoDataset(cfg, 'val')
print(dataset.__len__())

for i in range(10):
    print(f"img shape[{i}]: {dataset[i][0].shape}")
sample = dataset.__getitem__(0)
print(sample)

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=1):
    """Visualizes a single bounding box on the image"""
    #     x_min, y_min, w, h = bbox
    #     x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    x_min, x_max, y_min, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids):
    if type(image) == torch.Tensor:
        image = image.numpy().transpose(1, 2, 0)
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = str(int(category_id))
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)