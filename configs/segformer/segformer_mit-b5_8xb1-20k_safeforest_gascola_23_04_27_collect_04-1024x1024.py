_base_ = ['./segformer_mit-b0_8xb1-20k_safeforest_gascola_23_04_27_collect_04-1024x1024.py']

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b5.pth'),
        embed_dims=64,
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512], num_classes=15))
