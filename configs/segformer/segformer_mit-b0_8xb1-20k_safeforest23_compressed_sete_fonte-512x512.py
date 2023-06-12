_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/safeforest23_compressed_sete_fonte.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(type='SegDataPreProcessor',
    mean=[104.49391897563062, 106.03779528250102, 103.2223719219133],
    std=[63.32101522036137, 64.53359365946275, 68.28525305914673],
    crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b0.pth')),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(768, 768),
            decode_head=dict(
           loss_decode=dict(
               type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
               # DeepLab used this class weight for cityscapes
               class_weight=[0.84947903, 0.49437028, 0.77871951, 0.87743118]
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

train_dataloader = dict(batch_size=2, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
