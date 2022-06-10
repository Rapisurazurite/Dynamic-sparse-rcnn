import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sparse_rcnn.dataloader import CocoDataset
from sparse_rcnn.dataloader.dataset import build_coco_transforms
from sparse_rcnn.model import SparseRCNN, build_model
from sparse_rcnn.utils.config import cfg_from_yaml_file, cfg, cfg_from_list


def parse_args():
    parser = argparse.ArgumentParser(description="Train sparse rcnn")
    parser.add_argument("--dataset", type=str, default="sparse_rcnn/configs/coco.yaml")
    parser.add_argument("--model", type=str, default="sparse_rcnn/configs/sparse_rcnn.yaml")
    parser.add_argument("--extra_tag", type=str, default="default")
    parser.add_argument("--extern_callback", type=str, default=None)
    parser.add_argument("--max_checkpoints", type=int, default=5)
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    args = parser.parse_args()
    cfg_from_yaml_file(args.dataset, cfg)
    cfg_from_yaml_file(args.model, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def load_checkpoint(model, optimizer, ckpt_dir, logger):
    checkpoint_files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if len(checkpoint_files) != 0:
        checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[-1]))
        last_ckpt_file = checkpoint_files[-1]
        if logger is not None:
            logger.info("Loading checkpoint from %s", last_ckpt_file)
        state_dict = torch.load(last_ckpt_file, map_location=torch.device("cpu"))
        cur_epoch, cur_it = state_dict["epoch"] + 1, state_dict["it"]  # +1 because we want to start from next epoch
        model.load_state_dict(state_dict["model_state"])
        if optimizer is not None:
            optimizer.load_state_dict(state_dict["optimizer_state"])
        return cur_epoch, cur_it
    else:
        if logger is not None:
            logger.info("No checkpoint found in %s", ckpt_dir)
        return 0, 0


def prepare_data(img):
    pixel_mean = torch.Tensor(cfg.NORMALIZATION.PIXEL_MEAN).view(3, 1, 1)
    pixel_std = torch.Tensor(cfg.NORMALIZATION.PIXEL_STD).view(3, 1, 1)
    size_divisibility = 32

    img = [(_img - pixel_mean) / pixel_std for _img in img]
    image_sizes = torch.Tensor([[im.shape[-2], im.shape[-1]] for im in img])
    max_size = image_sizes.max(0)[0].int().tolist()

    if size_divisibility > 1:
        stride = size_divisibility
        max_size = [(d + (stride - 1)) // stride * stride for d in max_size]

    if len(img) == 1:
        image_size = image_sizes[0].numpy().astype(np.int)
        padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
        batched_imgs = F.pad(img[0], padding_size, value=0.0).unsqueeze_(0)
    else:
        batch_shape = [len(img)] + list(img[0].shape[:-2]) + list(max_size)
        batched_imgs = img[0].new_full(batch_shape, 0.0)
        for img, pad_img in zip(img, batched_imgs):
            pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

    return batched_imgs.contiguous()


def main():
    args, cfg = parse_args()
    transform = build_coco_transforms(cfg, mode="val")
    dataset = CocoDataset(cfg, mode="val", transform=transform)
    # dataloader = build_dataloader(cfg, transform, batch_size=1, dist=False, workers=0, mode="val")
    # model = SparseRCNN(
    #     cfg,
    #     num_classes=cfg.MODEL.SparseRCNN.NUM_CLASSES,
    #     backbone=cfg.MODEL.BACKBONE
    # )
    model = build_model(
        cfg,
        num_classes=cfg.MODEL.SparseRCNN.NUM_CLASSES,
        backbone=cfg.MODEL.BACKBONE
    )
    ckpt_dir = "./output/default/ckpt"
    load_checkpoint(model, optimizer=None, ckpt_dir=ckpt_dir, logger=None)
    idx = 0
    model.eval()
    for i in range(len(dataset)):
        img, img_hwhw, img_info = dataset[i]
        image_tensor = img.unsqueeze(0)
        image_tensor = prepare_data(image_tensor)
        img_hwhw = img_hwhw.unsqueeze(0)

        img_pred = img.numpy().transpose(1, 2, 0)[..., ::-1].copy()
        img_gt = img_pred.copy()

        output = model(image_tensor, img_hwhw)
        # TODO: change boxes to bboxes
        scores, labels, bboxes = output["scores"], output["labels"], output["boxes"]
        scores, labels, bboxes = scores[0], labels[0], bboxes[0]
        n_proposals = scores.shape[0]
        top_10_score = scores.topk(10, dim=0)[0][-1]
        for i in range(n_proposals):
            if scores[i] > top_10_score:
                box = bboxes[i].cpu().data.numpy().astype(np.int32)
                cv2.rectangle(img_pred, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        for box in img_info["gt_boxes"]:
            box = box.cpu().data.numpy().astype(np.int32)
            cv2.rectangle(img_gt, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

        cv2.imshow("pred", img_pred)
        cv2.imshow("gt", img_gt)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
