import argparse
import datetime
import glob
import os
import subprocess
import time
from typing import Dict, Any, List

import torch
import tqdm
from sparse_rcnn.utils.config import cfg_from_yaml_file, cfg, cfg_from_list, log_config_to_file
from sparse_rcnn.utils import common_utils
from sparse_rcnn.dataloader import build_dataloader
from sparse_rcnn.dataloader.dataset import build_coco_transforms
from sparse_rcnn.model import SparseRCNN
from sparse_rcnn.loss import SparseRcnnLoss
from sparse_rcnn.solver.__init__ import build_optimizer, build_lr_scheduler
from sparse_rcnn.evaluation.coco_evaluation import COCOEvaluator

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


def train_model(model, criterion, optimizer, train_loader, test_loader, scheduler, evaluator, start_epoch, total_epochs, device, logger,
                ckpt_save_dir, args, solver_cfg, extern_callback=None):
    model.train()
    with tqdm.trange(start_epoch, total_epochs, desc="epochs", dynamic_ncols=False) as ebar:
        for cur_epoch in ebar:
            # train_one_epoch(model, optimizer, criterion, train_loader, scheduler, cur_epoch, device, ebar, logger)
            # model_state = checkpoint_state(model=model, optimizer=optimizer, epoch=cur_epoch)
            # save_checkpoint(model_state, os.path.join(ckpt_save_dir, "checkpoint_epoch_%d" % (cur_epoch + 1)),
            #                 max_checkpoints=args.max_checkpoints)
            # logger.info("Saving checkpoint to %s\n", ckpt_save_dir)
            eval(evaluator, model, test_loader, cur_epoch=cur_epoch, device=device, logger=logger)
            if extern_callback is not None:
                try:
                    p = subprocess.Popen(extern_callback, shell=True)
                    p.wait()
                except Exception as e:
                    logger.error(e)
        pass


def eval(evaluator, model, test_loader, cur_epoch, device, logger):
    logger.info("Evaluating checkpoint at epoch %d", cur_epoch)
    model.eval()

    total_it_each_epoch = len(test_loader)
    dataloader_iter = iter(test_loader)

    tbar = tqdm.trange(total_it_each_epoch, desc="evaluating", dynamic_ncols=False)
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
                img_factor = torch.Tensor([label_sample["width"], label_sample["height"], label_sample["width"], label_sample["height"]]).to(device) / label_sample["image_size_xyxy"]
                output['boxes'][each_sample] *= img_factor

            evaluator.process(label, output)
            tbar.update()

            if cur_iter >= 300:
                break

        ret = evaluator.evaluate()
        logger.info("Evaluation result:")
        logger.info('AP : {AP:.3f}, AP50 : {AP50:.3f}, AP75 : {AP75:.3f}'.format(
            AP=ret["bbox"]["AP"],
            AP50=ret["bbox"]["AP50"],
            AP75=ret["bbox"]["AP75"]))
        return ret


def train_one_epoch(model, optimizer, criterion, train_loader, scheduler, cur_epoch, device, ebar, logger):
    model.train()

    total_it_each_epoch = len(train_loader)
    dataloader_iter = iter(train_loader)

    total_loss = common_utils.WindowAverageMeter()
    loss_ce = common_utils.WindowAverageMeter()
    loss_giou = common_utils.WindowAverageMeter()
    loss_bbox = common_utils.WindowAverageMeter()

    tbar = tqdm.trange(total_it_each_epoch, desc="train", dynamic_ncols=False)
    for cur_iter in range(total_it_each_epoch):
        end = time.time()
        batch = next(dataloader_iter)
        img, img_whwh, label = batch
        img, img_whwh = img.to(device), img_whwh.to(device)
        for t in label:
            for k in t.keys():
                if k in ['gt_boxes', 'gt_classes', 'image_size_xyxy', 'image_size_xyxy_tgt']:
                    t[k] = t[k].to(device)

        data_time = time.time()
        cur_data_time = data_time - end

        scheduler.step(cur_epoch * total_it_each_epoch + cur_iter)
        optimizer.zero_grad()

        output = model(img, img_whwh)
        loss: Dict[str, Any] = criterion(output, label)
        weighted_loss = loss["weighted_loss"]
        forward_timer = time.time()
        cur_forward_time = forward_timer - data_time

        weighted_loss.backward()
        optimizer.step()
        cur_batch_time = time.time() - end

        # --------------- display ---------------
        total_loss.update(weighted_loss.item())
        loss_ce.update(loss["loss_ce"].item())
        loss_giou.update(loss["loss_giou"].item())
        loss_bbox.update(loss["loss_bbox"].item())

        e_disp = {
            "lr": float(scheduler.get_lr()[0]),
            "dt": cur_data_time,
            "bt": cur_batch_time,
            "ft": cur_forward_time,
        }
        ebar.set_postfix(e_disp)
        ebar.refresh()
        disp_dict = {
            "l": total_loss.avg,
            "l_ce": loss_ce.avg,
            "l_giou": loss_giou.avg,
            "l_bbox": loss_bbox.avg,
        }
        tbar.set_postfix(disp_dict)
        tbar.update()
    # --------------- after train one epoch ---------------
    logger.info("Epoch %d, loss: %.4f, loss_ce: %.4f, loss_giou: %.4f, loss_bbox: %.4f",
                cur_epoch + 1, total_loss.all_avg, loss_ce.all_avg, loss_giou.all_avg, loss_bbox.all_avg)


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        model_state = model.state_dict()
    else:
        model_state = None
    return {
        "epoch": epoch,
        "it": it,
        "model_state": model_state,
        "optimizer_state": optim_state,
    }


def save_checkpoint(state: Dict[str, Any], filename="checkpoint", max_checkpoints=5):
    filename = f"{filename}.pth"
    torch.save(state, filename)
    save_path = os.path.dirname(filename)
    checkpoint_files = glob.glob(os.path.join(save_path, "checkpoint_epoch_*.pth"))
    checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[-1]))

    # remove old checkpoints if number of checkpoints exceed max_checkpoints
    if len(checkpoint_files) > max_checkpoints:
        for f in checkpoint_files[:-max_checkpoints]:
            os.remove(f)


def load_checkpoint(model, optimizer, ckpt_dir, logger):
    checkpoint_files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if len(checkpoint_files) != 0:
        checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[-1]))
        last_ckpt_file = checkpoint_files[-1]
        logger.info("Loading checkpoint from %s", last_ckpt_file)
        state_dict = torch.load(last_ckpt_file, map_location=torch.device("cpu"))
        cur_epoch, cur_it = state_dict["epoch"] + 1, state_dict["it"]  # +1 because we want to start from next epoch
        model.load_state_dict(state_dict["model_state"])
        optimizer.load_state_dict(state_dict["optimizer_state"])
        return cur_epoch, cur_it
    else:
        logger.info("No checkpoint found in %s", ckpt_dir)
        return 0, 0


def freeze_params_contain_keyword(model, keywords: List[str], logger):
    if keywords is None or len(keywords) == 0:
        return

    logger.info("Freezing params containing keywords: %s", keywords)
    for name, param in model.named_parameters():
        for keyword in keywords:
            if keyword in name:
                param.requires_grad = False
                logger.info("Freeze parameter %s", name)


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
    train_dataloader = build_dataloader(cfg,
                                        transforms=build_coco_transforms(cfg, mode="train"),
                                        batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                        dist=False,
                                        workers=4,
                                        mode="train")

    test_dataloader = build_dataloader(cfg,
                                   transforms=build_coco_transforms(cfg, mode="val"),
                                   batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                   dist=False,
                                   workers=0,
                                   mode="val")


    model = SparseRCNN(
        cfg,
        num_classes=cfg.MODEL.SparseRCNN.NUM_CLASSES,
        backbone="resnet18"
    )

    logger.info("Model: \n{}".format(model))

    model.to(device)
    criterion = SparseRcnnLoss(cfg)
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)
    evaluator = COCOEvaluator(cfg.BASE_ROOT, 'coco_2017_val', logger)

    start_epoch, cur_it = load_checkpoint(model, optimizer, ckpt_dir, logger)

    # freeze_params_contain_keyword(model, keywords=["backbone"], logger=logger)

    train_model(model,
                criterion,
                optimizer,
                train_loader=train_dataloader,
                test_loader=test_dataloader,
                scheduler=lr_scheduler,
                evaluator=evaluator,
                start_epoch=start_epoch,
                total_epochs=cfg.SOLVER.NUM_EPOCHS,
                device=device,
                logger=logger,
                ckpt_save_dir=ckpt_dir,
                args=args,
                solver_cfg=cfg.SOLVER,
                extern_callback=args.extern_callback)


if __name__ == "__main__":
    main()
