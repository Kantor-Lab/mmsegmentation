
# dataset settings
dataset_type = "Safeforest23Dataset"
# dataset settings
#data_root = "/ofo-share/repos-david/data/mmseg-training/safeforest23"
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
# Individual datasets
dataset_train_gascola_4 = dict(
    type="Safeforest23Dataset",
    data_root="/ofo-share/repos-david/Safeforest_CMU_data_dvc//data/site_Gascola/04_27_23/collect_04/processed_01/annotations/safeforest23",
    data_prefix=dict(
        img_path='img_dir/train', seg_map_path='ann_dir/train'),
    pipeline=train_pipeline)

dataset_val_gascola_4 = dict(
    type="Safeforest23Dataset",
    data_root="/ofo-share/repos-david/Safeforest_CMU_data_dvc//data/site_Gascola/04_27_23/collect_04/processed_01/annotations/safeforest23",
    data_prefix=dict(
        img_path='img_dir/val', seg_map_path='ann_dir/val'),
    pipeline=test_pipeline)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type="ConcatDataset",
        datasets=[dataset_train_gascola_4]))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[dataset_val_gascola_4])
    )
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator