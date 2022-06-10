import argparse
import datetime
import os
import subprocess
import time
from typing import Dict, Any

import torch
import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from sparse_rcnn.dataloader import build_dataloader
from sparse_rcnn.dataloader.dataset import build_coco_transforms
from sparse_rcnn.evaluation.coco_evaluation import COCOEvaluator
from sparse_rcnn.loss import SparseRcnnLoss
from sparse_rcnn.model import SparseRCNN, build_model
from sparse_rcnn.solver import build_optimizer, build_lr_scheduler
from sparse_rcnn.utils import common_utils, commu_utils
from sparse_rcnn.utils.config import cfg_from_yaml_file, cfg, cfg_from_list, log_config_to_file
from sparse_rcnn.utils.train_utils import checkpoint_state, save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train sparse rcnn")

    # Base args
    parser.add_argument("--dataset", type=str, default="sparse_rcnn/configs/coco.yaml")
    parser.add_argument("--model", type=str, default="sparse_rcnn/configs/sparse_rcnn.yaml")
    parser.add_argument("--extra_tag", type=str, default="default", help="extra tag for model saving")
    parser.add_argument("--extern_callback", type=str, default=None, help="when a epoch is done, run this command")
    parser.add_argument("--max_checkpoints", type=int, default=5, help="maximum number of checkpoints to keep")
    parser.add_argument("--log_iter", type=int, default=200, help="log the model info every N iterations")

    # fp16 mixed precision
    parser.add_argument("--fp16_mix", action="store_true", help="use fp16 mixed precision")

    # Multi-GPU setting
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')

    # Modified the configs by args
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    args = parser.parse_args()
    cfg_from_yaml_file(args.dataset, cfg)
    cfg_from_yaml_file(args.model, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def train_model(model, criterion, optimizer, evaluator, train_loader, test_loader, scheduler, start_epoch, total_epochs,
                device, logger, ckpt_save_dir, args, cfg, extern_callback=None):
    scaler = GradScaler() if args.fp16_mix else None
    model.train()
    with tqdm.trange(start_epoch, total_epochs, desc="epochs", ncols=120) as ebar:
        for cur_epoch in ebar:

            train_one_epoch(model, criterion, optimizer, train_loader, scheduler, cur_epoch, device, logger, args, cfg,
                            ebar, scaler)
            if cfg.LOCAL_RANK == 0:
                model_state = checkpoint_state(model=model, optimizer=optimizer, epoch=cur_epoch)
                save_checkpoint(model_state, os.path.join(ckpt_save_dir, "checkpoint_epoch_%d" % cur_epoch),
                                max_checkpoints=args.max_checkpoints)
                logger.info("Saving checkpoint to %s\n", ckpt_save_dir)

                # currenly only support eval on single card
                eval(evaluator, model, test_loader, cur_epoch=cur_epoch, device=device, logger=logger, args=args,
                     cfg=cfg)
            if extern_callback is not None and cfg.LOCAL_RANK == 0:
                try:
                    p = subprocess.Popen(extern_callback, shell=True)
                    p.wait()
                except Exception as e:
                    logger.error(e)


def eval(evaluator, model, test_loader, cur_epoch, device, logger, args, cfg):
    logger.info("Evaluating checkpoint at epoch %d", cur_epoch)
    model.eval()

    total_it_each_epoch = len(test_loader)
    dataloader_iter = iter(test_loader)

    tbar = tqdm.trange(total_it_each_epoch, desc="evaluating", ncols=120)
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


def train_one_epoch(model, criterion, optimizer, train_loader, scheduler, cur_epoch, device, logger, args, cfg, ebar, scaler):
    model.train()

    total_it_each_epoch = len(train_loader)
    dataloader_iter = iter(train_loader)

    if cfg.LOCAL_RANK == 0:
        data_timer = common_utils.AverageMeter()
        batch_timer = common_utils.AverageMeter()
        forward_timer = common_utils.AverageMeter()

    tbar = tqdm.trange(total_it_each_epoch, desc="train", ncols=120)
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

        if args.fp16_mix:
            with autocast():
                output = model(img, img_whwh)
                loss: Dict[str, Any] = criterion(output, label)
        else:
            output = model(img, img_whwh)
            loss: Dict[str, Any] = criterion(output, label)

        weighted_loss = loss["weighted_loss"]
        forward_time = time.time()
        cur_forward_time = forward_time - data_time

        if args.fp16_mix:
            scaler.scale(weighted_loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            weighted_loss.backward()
            optimizer.step()
        cur_batch_time = time.time() - end

        # average in different ranks
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        if cfg.LOCAL_RANK == 0:
            # --------------- display ---------------
            data_timer.update(avg_data_time)
            batch_timer.update(avg_batch_time)
            forward_timer.update(avg_forward_time)

            e_disp = {
                "lr": float(scheduler.get_lr()[0]),
                "d_time": f"{data_timer.val:.2f}/{data_timer.avg:.2f}",
                "f_time": f"{forward_timer.val:.2f}/{forward_timer.avg:.2f}",
                "b_time": f"{batch_timer.val:.2f}/{batch_timer.avg:.2f}",
            }
            ebar.set_postfix(e_disp)
            ebar.refresh()
            t_disp = {
                "l": f"{weighted_loss.item():.2f}",
                "l_ce": f"{loss['loss_ce'].item():.2f}",
                "l_giou": f"{loss['loss_giou'].item():.2f}",
                "l_bbox": f"{loss['loss_bbox'].item():.2f}",
            }
            tbar.set_postfix(t_disp)
            tbar.update()

            # ----------------- log -----------------
            if cur_iter % args.log_iter == 0:
                logger.info(
                    f"\nEpoch: [{cur_epoch}][{cur_iter}/{total_it_each_epoch}], \n lr:{e_disp['lr']}, loss: {t_disp['l']}, loss_ce: {t_disp['l_ce']}, loss_giou: {t_disp['l_giou']}, loss_bbox: {t_disp['l_bbox']}")
                # TODO: debug
                proposaL_box = model.dynamic_proposal_generator.init_proposal_boxes.weight.data.cpu().numpy()
                logger.info(f"proposaL_box: {proposaL_box}")


    # --------------- after train one epoch ---------------
    logger.info("Epoch: {} finished!".format(cur_epoch))
    logger.info(
        f"Epoch: {cur_epoch}, \n lr:{e_disp['lr']}, loss: {t_disp['l']}, loss_ce: {t_disp['l_ce']}, loss_giou: {t_disp['l_giou']}, loss_bbox: {t_disp['l_bbox']}")


def main():
    args, cfg = parse_args()

    # --------------- Multi-GPU setting ---------------
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    # --------------- save dir ---------------
    output_dir = os.path.join("./output", args.extra_tag, "results")
    ckpt_dir = os.path.join("./output", args.extra_tag, "ckpt")
    log_file = os.path.join(output_dir, "log_train_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # --------------- logger ---------------
    logger = common_utils.create_logger(log_file=log_file, rank=cfg.LOCAL_RANK)
    device = torch.device(cfg.DEVICE)
    logger.info('**********************Start logging**********************')
    log_config_to_file(cfg, logger=logger)

    # ------------ Create dataloader ------------
    train_dataloader = build_dataloader(cfg,
                                        transforms=build_coco_transforms(cfg, mode="train"),
                                        batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                        dist=dist_train,
                                        workers=2,
                                        pin_memory=True,
                                        mode="train")
    test_loader = build_dataloader(cfg,
                                   transforms=build_coco_transforms(cfg, mode="val"),
                                   batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                   dist=dist_train,
                                   workers=2,
                                   pin_memory=False,
                                   mode="val")

    # --------------- Create model ---------------
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

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    logger.info("Model: \n{}".format(model))

    model.to(device)
    criterion = SparseRcnnLoss(cfg)
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)
    evaluator = COCOEvaluator(cfg.BASE_ROOT, 'coco_2017_val', logger)

    start_epoch, cur_it = load_checkpoint(model, optimizer, ckpt_dir, logger)

    # freeze_params_contain_keyword(model, keywords=["backbone"], logger=logger)

    if dist_train:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()],
            find_unused_parameters=True
        )

    train_model(model,
                criterion,
                optimizer,
                evaluator=evaluator,
                train_loader=train_dataloader,
                test_loader=test_loader,
                scheduler=lr_scheduler,
                start_epoch=start_epoch,
                total_epochs=cfg.SOLVER.NUM_EPOCHS,
                device=device,
                logger=logger,
                ckpt_save_dir=ckpt_dir,
                args=args,
                cfg=cfg,
                extern_callback=args.extern_callback)


if __name__ == "__main__":
    main()
