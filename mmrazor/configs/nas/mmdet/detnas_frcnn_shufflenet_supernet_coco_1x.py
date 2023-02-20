_base_ = [
    '..._base_.nas_backbones.spos_shufflenet_supernet',
    'mmdet.configs._base_.coco_detection',
    'mmdet.configs._base_.default_runtime',
    'mmdet.configs._base_.faster_rcnn_r50',
    'mmdet.configs.faster_rcnn.faster_rcnn_r50_fpn_1x_coco',
    'mmdet.configs._base_.scheduler_1x'
]

from mmdet.configs._base_.coco_detection import *
from mmdet.configs._base_.default_runtime import *
from mmdet.configs._base_.scheduler_1x import *
from mmdet.configs.faster_rcnn.faster_rcnn_r50_fpn_1x_coco import *
from mmdet.models import Shared4Conv1FCBBoxHead

from mmrazor.models import SPOS, OneShotModuleMutator
from ..._base_.nas_backbones.spos_shufflenet_supernet import *

norm_cfg = dict(type='SyncBN', requires_grad=True)

model.backbone = nas_backbone
model.backbone.norm_cfg = norm_cfg
model.backbone.out_indices = (0, 1, 2, 3)
model.backbone.with_last_layer = False

model.neck.norm_cfg = norm_cfg
model.neck.in_channels = [64, 160, 320, 640]


model.roi_head.bbox_head.norm_cfg = norm_cfg
model.roi_head.bbox_head = Shared4Conv1FCBBoxHead(
    in_channels=256,
    fc_out_channels=1024,
    roi_feat_size=7,
    num_classes=80,
    bbox_coder=DeltaXYWHBBoxCoder(
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2]),
    reg_class_agnostic=False,
    loss_cls=CrossEntropyLoss(use_sigmoid=False),
    loss_bbox=L1Loss(loss_weight=1.0)
)

model = SPOS(
    architecture=model,
    mutator=OneShotModuleMutator()
)


from mmengine.optim import LinearLR, MultiStepLR

param_scheduler = [
    LinearLR.build_iter_from_epoch(optimizer='???', start_factor=0.001, by_epoch=False, begin=0, end=500),
    MultiStepLR.build_iter_from_epoch(optimizer='???', begin=0, end=12, by_epoch=True, milestones=[8, 11], gamma=0.1)
]

# model.train()
# model = model.init_weights()
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=12,
#         by_epoch=True,
#         milestones=[8, 11],
#         gamma=0.1)
# ]

find_unused_parameters = True
result = runner.train()
