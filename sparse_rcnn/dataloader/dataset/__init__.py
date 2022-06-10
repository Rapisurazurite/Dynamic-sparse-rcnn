from .coco import CocoDataset
from .coco_transform import SmallestMaxSize_v2
import albumentations as A


# def build_coco_transforms(cfg, mode="train"):
#     assert mode in ["train", "val"], "Unknown mode '{}'".format(mode)
#     min_size = cfg.INPUT.MIN_SIZE_TRAIN
#     max_size = cfg.INPUT.MAX_SIZE_TRAIN
#     sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
#     if mode == "train":
#         coco_train_transform = Compose([
#             RandomFlip(),
#             ResizeShortestEdge(min_size, max_size, sample_style),
#         ])
#         return coco_train_transform
#     elif mode == "val":
#         coco_val_transform = Compose([
#             ResizeShortestEdge(min_size, max_size, sample_style),
#         ])
#         return coco_val_transform

def build_coco_transforms(cfg, mode="train"):
    assert mode in ["train", "val"], "Unknown mode '{}'".format(mode)
    if mode == "train":
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        coco_train_transform = A.Compose([
            SmallestMaxSize_v2(max_size=min_size, max_limit=max_size, p=1),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=30, p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.2)],
            # x_min, y_min, x_max, y_max
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=["classes"])
        )
        return coco_train_transform
    elif mode == "val":
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        coco_val_transform = A.Compose([
            SmallestMaxSize_v2(max_size=min_size, max_limit=max_size, p=1)],
            # x_min, y_min, x_max, y_max
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=["classes"])
        )
        return coco_val_transform
