from torch.utils.data import DataLoader

from .collate import Collate
from .dataset import CocoDataset
from .sampler import RandomSampler, AspectRatioBasedSampler, DistributedGroupSampler

__all__ = {
    "CocoDataset": CocoDataset
}


def build_dataloader(dataset_cfg, transforms, batch_size, dist, workers=4, pin_memory=True,
                     mode="train"):
    if dataset_cfg.DATASET not in __all__.keys():
        raise ValueError("Dataset {} not supported".format(dataset_cfg.DATASET))

    dataset = __all__[dataset_cfg.DATASET](dataset_cfg, mode, transforms)

    if mode == "train":
        if dist:
            # sampler = DistributedSampler(dataset, shuffle=(mode == "train"))
            sampler = DistributedGroupSampler(dataset, samples_per_gpu=batch_size)
        else:
            # Note: Use this sampler to reduce memory usage.
            sampler = AspectRatioBasedSampler(dataset, batch_size=batch_size, drop_last=True)

        dataloader = DataLoader(
            dataset,
            pin_memory=pin_memory,
            num_workers=workers,
            collate_fn=Collate(dataset_cfg),
            batch_sampler=sampler,
        )
    elif mode == "val":
        if dist:
            raise ValueError("DDP currently does not support validation")
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                pin_memory=pin_memory,
                num_workers=workers,
                shuffle=False,
                collate_fn=Collate(dataset_cfg),
                drop_last=False
            )
    else:
        raise ValueError("Mode {} not supported".format(mode))
    return dataloader
