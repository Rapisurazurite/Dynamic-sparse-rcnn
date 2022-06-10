from sparse_rcnn.utils.config import cfg_from_yaml_file, cfg
from sparse_rcnn.dataloader.dataset import build_coco_transforms
import argparse

from sparse_rcnn.dataloader import build_dataloader

parser = argparse.ArgumentParser(description="Test coco dataset")
coco_config = "sparse_rcnn/configs/coco.yaml"
cfg_from_yaml_file(coco_config, cfg)

transforms = build_coco_transforms(cfg, mode="train")
dataloader = build_dataloader(cfg, transforms, batch_size=2, dist=False, workers=0, mode="train")

for i, data in enumerate(dataloader):
    # print(data)
    img, img_whwh, label = data
    print(f"img_shape: {img.shape}")
    print(f"img_whwh: {img_whwh}")
    print(f"label: {label}")
    break
print("Done")
