import glob
import os
from typing import Dict, Any, List

import torch


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
    # if specified the ckpt file
    if os.path.isfile(ckpt_dir):
        logger.info("Loading checkpoint from %s", ckpt_dir)
        state_dict = torch.load(ckpt_dir, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict["model_state"])
        return 0, 0
    # or specified the ckpt dir
    else:
        checkpoint_files = glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.pth"))
        # load the latest checkpoint
        if len(checkpoint_files) != 0:
            checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[-1]))
            last_ckpt_file = checkpoint_files[-1]
            logger.info("Loading checkpoint from %s", last_ckpt_file)
            state_dict = torch.load(last_ckpt_file, map_location=torch.device("cpu"))
            cur_epoch, cur_it = state_dict["epoch"] + 1, state_dict["it"]  # +1 because we want to start from next epoch
            model.load_state_dict(state_dict["model_state"])
            if optimizer is not None:
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
