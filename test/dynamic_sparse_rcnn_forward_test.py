import argparse

from sparse_rcnn.utils.config import cfg_from_yaml_file, cfg
from sparse_rcnn.dataloader.dataset import build_coco_transforms
from sparse_rcnn.dataloader import build_dataloader
from sparse_rcnn.model import build_model
from sparse_rcnn.evaluation.coco_evaluation import COCOEvaluator


parser = argparse.ArgumentParser(description="Test coco dataset")
coco_config = "../sparse_rcnn/configs/coco.yaml"
model_config = "../sparse_rcnn/configs/dynamic_sparse_rcnn.yaml"
cfg_from_yaml_file(coco_config, cfg)
cfg_from_yaml_file(model_config, cfg)

# input = torch.randn(*[2, 3, 800, 1216])
# img_whwh = torch.tensor([[721, 480, 721, 480],
#                          [800, 1216, 800, 1216]])

val_transform = build_coco_transforms(cfg, mode="val")
dataloader, _ = build_dataloader(cfg, val_transform, batch_size=2, dist=False, workers=0, mode="val")
model = build_model(cfg, num_classes=80, backbone="resnet18")
model.train()

for i, data in enumerate(dataloader):
    img, img_whwh, label = data
    print(f"img_shape: {img.shape}")
    print(f"img_whwh: {img_whwh}")
    print(f"label: {label}")

    output = model(img, img_whwh)

    for name, value in output.items():
        if name != "aux_outputs":
            print(f"{name}: {value.shape}")
    break