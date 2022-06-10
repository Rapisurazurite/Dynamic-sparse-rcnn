import argparse

from sparse_rcnn.utils.config import cfg_from_yaml_file, cfg
from sparse_rcnn.dataloader.dataset import CocoDataset


parser = argparse.ArgumentParser(description="Test coco dataset")
coco_config = "../sparse_rcnn/configs/coco.yaml"
cfg_from_yaml_file(coco_config, cfg)


dataset = CocoDataset(cfg, 'val')
print(dataset.__len__())

for i in range(10):
    print(f"img shape[{i}]: {dataset[i][0].shape}")
sample = dataset.__getitem__(0)
print(sample)