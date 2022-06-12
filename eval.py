import argparse
import datetime
import glob
import os

import torch
import tqdm

from sparse_rcnn.dataloader import build_dataloader
from sparse_rcnn.dataloader.dataset import build_coco_transforms
from sparse_rcnn.evaluation.coco_evaluation import COCOEvaluator
from sparse_rcnn.model import SparseRCNN, build_model
from sparse_rcnn.utils import common_utils
from sparse_rcnn.utils.config import cfg_from_yaml_file, cfg, cfg_from_list, log_config_to_file
from sparse_rcnn.utils.train_utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train sparse rcnn")
    parser.add_argument("--dataset", type=str, default="sparse_rcnn/configs/coco.yaml")
    parser.add_argument("--model", type=str, default="sparse_rcnn/configs/sparse_rcnn.yaml")
    parser.add_argument("--extra_tag", type=str, default="default")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    args = parser.parse_args()
    cfg_from_yaml_file(args.dataset, cfg)
    cfg_from_yaml_file(args.model, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval(evaluator, model, test_loader, cur_epoch, device, logger):
    logger.info("Evaluating checkpoint at epoch %d", cur_epoch)
    model.eval()

    total_it_each_epoch = len(test_loader)
    dataloader_iter = iter(test_loader)

    tbar = tqdm.trange(total_it_each_epoch, desc="evaluating", ncols=80)
    evaluator.reset()
    with torch.no_grad():
        for cur_iter in range(total_it_each_epoch):
            batch = next(dataloader_iter)
            img, img_whwh, label = batch
            img, img_whwh = img.to(device), img_whwh.to(device)
            for t in label:
                for k in t.keys():
                    if k in ['gt_boxes', 'gt_classes', 'image_size_xyxy', 'image_size_xyxy_tgt']:
                        t[k] = t[k].to(device)

            output = model(img, img_whwh)
            batch_size = img.shape[0]

            for each_sample in range(batch_size):
                label_sample = label[each_sample]
                img_factor = torch.Tensor(
                    [label_sample["width"], label_sample["height"], label_sample["width"], label_sample["height"]]).to(
                    device) / label_sample["image_size_xyxy"]
                output['boxes'][each_sample] *= img_factor

            evaluator.process(label, output)
            tbar.update()


        ret = evaluator.evaluate()
        logger.info("Evaluation result:")
        logger.info('AP : {AP:.3f}, AP50 : {AP50:.3f}, AP75 : {AP75:.3f}'.format(
            AP=ret["bbox"]["AP"],
            AP50=ret["bbox"]["AP50"],
            AP75=ret["bbox"]["AP75"]))
        return ret


def main():
    args, cfg = parse_args()
    output_dir = os.path.join("./output", args.extra_tag, "results")
    ckpt_dir = os.path.join("./output", args.extra_tag, "ckpt")
    log_file = os.path.join(output_dir, "log_train_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = common_utils.create_logger(log_file=log_file)
    device = torch.device(cfg.DEVICE)
    logger.info('**********************Start logging**********************')
    log_config_to_file(cfg, logger=logger)
    # ------------ Create dataloader ------------
    test_loader, _ = build_dataloader(cfg,
                                   transforms=build_coco_transforms(cfg, mode="val"),
                                   batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                   dist=False,
                                   workers=4,
                                   mode="val")

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

    logger.info("Model: \n{}".format(model))

    model.to(device)
    optimizer = None
    evaluator = COCOEvaluator(cfg.BASE_ROOT, cfg.DATASETS.TEST[0], logger)

    # if specified ckpt file, load it
    if args.ckpt:
        logger.info("You specified a ckpt file, loading it")
        start_epoch, cur_it = load_checkpoint(model, optimizer, args.ckpt, logger)
        eval(evaluator, model, test_loader, cur_epoch=start_epoch, device=device, logger=logger)
    # eval the ckpts in ckpt_dir
    else:
        logger.info("You did not specify a ckpt file, loading the models checkpoints in ckpt_dir")
        checkpoint_files = glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.pth"))
        checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[-1]), reverse=True)
        for ckpt_file in checkpoint_files:
            start_epoch, cur_it = load_checkpoint(model, optimizer, ckpt_file, logger)
            eval(evaluator, model, test_loader, cur_epoch=start_epoch, device=device, logger=logger)


if __name__ == "__main__":
    main()
