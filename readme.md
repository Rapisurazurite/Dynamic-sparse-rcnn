# Dynamic Sparse-RCNN inplementation 
This is an unofficial pytorch implementation of Dynamic Sparse RCNN object detection as described in [Dynamic Sparse R-CNN](https://arxiv.org/abs/2205.02101) by Qinghang Hong, Fengming Liu, Dong Li, Ji Liu, Lu Tian, Yi Shan.

Sparse R-CNN is a recent strong object detection baseline by set prediction on sparse, learnable proposal boxes and proposal features. In this work, we propose to improve Sparse R-CNN with two dynamic designs. First, Sparse R-CNN adopts a one-to-one label assignment scheme, where the Hungarian algorithm is applied to match only one positive sample for each ground truth. Such one-to-one assignment may not be optimal for the matching between the learned proposal boxes and ground truths. To address this problem, we propose dynamic label assignment (DLA) based on the optimal transport algorithm to assign increasing positive samples in the iterative training stages of Sparse R-CNN. We constrain the matching to be gradually looser in the sequential stages as the later stage produces the refined proposals with improved precision. Second, the learned proposal boxes and features remain fixed for different images in the inference process of Sparse R-CNN. Motivated by dynamic convolution, we propose dynamic proposal generation (DPG) to assemble multiple proposal experts dynamically for providing better initial proposal boxes and features for the consecutive training stages. DPG thereby can derive sample-dependent proposal boxes and features for inference. Experiments demonstrate that our method, named Dynamic Sparse R-CNN, can boost the strong Sparse R-CNN baseline with different backbones for object detection. Particularly, Dynamic Sparse R-CNN reaches the state-of-the-art 47.2% AP on the COCO 2017 validation set, surpassing Sparse R-CNN by 2.2% AP with the same ResNet-50 backbone.

There are still some bugs in the implementation that have not been fixed, making the use of DPG even worse.


## Roadmap
- [x] Use albumentations instead of the basic transforms
- [x] Add eval script and demo
- [x] fp16 mixed precision training
- [x] OTA assignment matcher
- [x] Unit increase strategy
- [ ] Dynamic k estimation
- [x] Dynamic Proposal Generation
- [ ] MAE
- [ ] Voc dataset support 
- [x] Support for multiple GPUs

## Example

- train on coco dataset with resnet50 backbone

    ```
    python train.py \
      --dataset sparse_rcnn/configs/coco.yaml \
      --model sparse_rcnn/configs/dynamic_sparse_rcnn.yaml \
      --set BASE_ROOT /home/input/coco-2017-dataset/coco2017 SOLVER.IMS_PER_BATCH 4 MODEL.BACKBONE "resnet50"
    ```

- train on coco dataset using fp16 mixed precision training with efficientnet_b3 backbone

    ```
    python train.py \
      --fp16_mix \
      --dataset sparse_rcnn/configs/coco.yaml \
      --model sparse_rcnn/configs/dynamic_sparse_rcnn.yaml \
      --set BASE_ROOT /home/input/coco-2017-dataset/coco2017 SOLVER.IMS_PER_BATCH 4 MODEL.BACKBONE "efficientnet_b3"
    ```
- you can specify the matcher: OtaMatcher or HungarianMatcher

    ```
    python train.py \
      --fp16_mix \
      --dataset sparse_rcnn/configs/coco.yaml \
      --model sparse_rcnn/configs/dynamic_sparse_rcnn.yaml \
      --set MODEL.LOSS.MATCHER.NAME HungarianMatcher
    ```
  
## Reference
```text
@article{hong2022dynamic,
  title={Dynamic Sparse R-CNN},
  author={Hong, Qinghang and Liu, Fengming and Li, Dong and Liu, Ji and Tian, Lu and Shan, Yi},
  journal={arXiv preprint arXiv:2205.02101},
  year={2022}
}
```
