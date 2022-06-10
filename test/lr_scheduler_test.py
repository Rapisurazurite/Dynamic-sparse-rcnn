import numpy as np
import tqdm
import env
import matplotlib.pyplot as plt

from sparse_rcnn.model import SparseRCNN
from sparse_rcnn.solver.__init__ import build_optimizer, build_lr_scheduler


from sparse_rcnn.utils.config import cfg_from_yaml_file, cfg

coco_config = "sparse_rcnn/configs/coco.yaml"
model_config = "sparse_rcnn/configs/sparse_rcnn.yaml"
cfg_from_yaml_file(coco_config, cfg)
cfg_from_yaml_file(model_config, cfg)

model = SparseRCNN(cfg, num_classes=81, backbone='resnet18')
optimizer = build_optimizer(cfg, model)
scheduler = build_lr_scheduler(cfg, optimizer)


# lr 从0->BASE_LR, iter 0->WARMUP_ITERS
# end = 2000
# step = np.arange(0, end, 1)
# lr = np.zeros(end)
# for i in tqdm.trange(end):
#     scheduler.step()
#     lr[i] = scheduler.get_lr()[0]
# plt.plot(step, lr)
# plt.show()

# lr = 0.1*lr, iter = 210000, 250000
# end = 300000
# step = np.arange(0, end, 1)
# lr = np.zeros(end)
# for i in tqdm.trange(end):
#     scheduler.step()
#     lr[i] = scheduler.get_lr()[0]
# plt.plot(step, lr)
# plt.show()

scheduler.step(500)
print(scheduler.get_lr()[0])
scheduler.step(1000)
print(scheduler.get_lr()[0])
scheduler.step(2000)
print(scheduler.get_lr()[0])
scheduler.step(500)
print(scheduler.get_lr()[0])