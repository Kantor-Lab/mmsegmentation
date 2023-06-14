_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/safeforest_gascola_23_04_27_collect_04.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (1024, 1024)
data_preprocessor = dict(type='SegDataPreProcessor',
    size=crop_size,
    mean=[71.02322328426814, 70.6379940200364, 71.67889280630872],
    std=[33.02850636286781, 32.481152762876164, 33.01011404576837])
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b0.pth')),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(768, 768),
            decode_head=dict(
           loss_decode=dict(
               type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
               # DeepLab used this class weight for cityscapes
               class_weight=[0.87789837, 0.98765434, 0.99733902, 0.98902638,
                0.84340816, 1., 0.99959211, 0.99971251, 0.99414109, 0.84302672,
                 1.,         1., 1.,         0.99067129, 1.]
            ))))


optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=20000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=1, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
