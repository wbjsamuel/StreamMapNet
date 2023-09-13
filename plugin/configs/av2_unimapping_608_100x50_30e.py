_base_ = [
    './_base_/default_runtime.py'
]

# model type
type = 'Mapper'
plugin = True

# plugin code dir
plugin_dir = 'plugin/'

# img configs
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

img_h = 640
img_w = 640
img_size = (img_h, img_w)

num_gpus = 8
batch_size = 4
num_iters_per_epoch = 24000 // (num_gpus * batch_size)
num_epochs = 30
total_iters = num_iters_per_epoch * num_epochs
num_iters_single_frame = total_iters // 6
num_queries = 100

# category configs
cat2id = {
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
}
num_class = max(list(cat2id.values())) + 1

# bev configs
roi_size = (100, 50)
bev_h = 50
bev_w = 100
pc_range = [-roi_size[0]/2, -roi_size[1]/2, -3, roi_size[0]/2, roi_size[1]/2, 5]

# vectorize params
coords_dim = 2
sample_dist = -1
sample_num = -1
simplify = True

class_names = ['lane_segment', 'ped_crossing']

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
num_cams = 7

map_size = [-30, -15, 30, 15]
para_method = 'downsample'
method_para = dict(n_points=10)
code_size = 3 * method_para['n_points'] * 3

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_ffn_cfg_ = dict(
    type='FFN',
    embed_dims=_dim_,
    feedforward_channels=_ffn_dim_,
    num_fcs=2,
    ffn_drop=0.1,
    act_cfg=dict(type='ReLU', inplace=True),
)
# meta info for submission pkl
meta = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
    output_format='vector')

# model configs
bev_embed_dims = 256
embed_dims = 512
num_feat_levels = 3
norm_cfg = dict(type='BN2d')
num_class = max(list(cat2id.values()))+1
num_points = 20
permute = True

model = dict(
    type='StreamMapNet',
    roi_size=roi_size,
    bev_h=bev_h,
    bev_w=bev_w,
    backbone_cfg=dict(
        type='BEVFormerBackbone',
        roi_size=roi_size,
        bev_h=bev_h,
        bev_w=bev_w,
        use_grid_mask=True,
        img_backbone=dict(
            type='ResNet',
            with_cp=False,
            # pretrained='./resnet50_checkpoint.pth',
            pretrained='open-mmlab://detectron2/resnet50_caffe',
            depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=-1,
            norm_cfg=norm_cfg,
            norm_eval=True,
            style='caffe',
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True)),
        img_neck=dict(
            type='FPN',
            in_channels=[512, 1024, 2048],
            out_channels=bev_embed_dims,
            start_level=0,
            add_extra_convs=True,
            num_outs=num_feat_levels,
            norm_cfg=norm_cfg,
            relu_before_extra_convs=True),
        transformer=dict(
            type='PerceptionTransformer',
            embed_dims=bev_embed_dims,
            num_cams=7,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=1,
                pc_range=pc_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=bev_embed_dims,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=bev_embed_dims,
                                num_points=8,
                                num_levels=num_feat_levels),
                            embed_dims=bev_embed_dims,
                            num_cams=7,
                        )
                    ],
                    feedforward_channels=bev_embed_dims*2,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                    'ffn', 'norm')
                )
            ),
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=bev_embed_dims//2,
            row_num_embed=bev_h,
            col_num_embed=bev_w,
            ),
    ),
    head_cfg=dict(
        type='MapDetectorHead',
        num_queries=num_queries,
        embed_dims=embed_dims,
        num_classes=num_class,
        in_channels=embed_dims//2,
        num_points=num_points,
        roi_size=roi_size,
        coord_dim=2,
        different_heads=False,
        predict_refine=False,
        sync_cls_avg_factor=True,
        streaming_cfg=dict(
            streaming=True,
            batch_size=batch_size,
            topk=int(num_queries*(1/3)),
            trans_loss_weight=0.1,
        ),
        transformer=dict(
            type='MapTransformer',
            num_feature_levels=1,
            num_points=num_points,
            coord_dim=2,
            encoder=dict(
                type='PlaceHolderEncoder',
                embed_dims=embed_dims,
            ),
            decoder=dict(
                type='MapTransformerDecoder_new',
                num_layers=6,
                return_intermediate=True,
                prop_add_stage=1,
                transformerlayers=dict(
                    type='MapTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            attn_drop=0.1,
                            proj_drop=0.1,
                        ),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            num_levels=1,
                            num_points=num_points,
                            dropout=0.1,
                        ),
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=embed_dims,
                        feedforward_channels=embed_dims*2,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),        
                    ),
                    feedforward_channels=embed_dims*2,
                    ffn_dropout=0.1,
                    # operation_order=('norm', 'self_attn', 'norm', 'cross_attn',
                    #                 'norm', 'ffn',)
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                    'ffn', 'norm')
                )
            )
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=5.0
        ),
        loss_reg=dict(
            type='LinesL1Loss',
            loss_weight=50.0,
            beta=0.01,
        ),
        assigner=dict(
            type='HungarianLinesAssigner',
                cost=dict(
                    type='MapQueriesCost',
                    cls_cost=dict(type='FocalLossCost', weight=5.0),
                    reg_cost=dict(type='LinesL1Cost', weight=50.0, beta=0.01, permute=permute),
                    ),
                ),
        ),
    streaming_cfg=dict(
        streaming_bev=True,
        batch_size=batch_size,
        fusion_cfg=dict(
            type='ConvGRU',
            out_channels=bev_embed_dims,
        )
    ),
    model_name='SingleStage'
)

# data processing pipelines
train_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='ResizeMultiViewImages',
         size=img_size, # H, W
         change_intrinsics=True,
         ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img', 'vectors'], meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'))
]

# data processing pipelines
test_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='ResizeMultiViewImages',
         size=img_size, # H, W
         change_intrinsics=True,
         ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img'], meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'))
]

# configs for evaluation code
# DO NOT CHANGE
dataset_type = 'AV2_UniMapping_Dataset'
data_root = 'datasets/openlane_v2_av2/'

eval_config = dict(
    type=dataset_type,
    ann_file= data_root + 'data_dict_subset_A_train_ls.pkl',
    scene_map_file=data_root + 'data_dict_av2_train_ls_v3_scene.pkl',
    meta=meta,
    roi_size=roi_size,
    cat2id=cat2id,
    pipeline=[
        dict(type='FormatBundleMap'),
        dict(type='Collect3D', keys=['vectors'], meta_keys=['token'])
    ],
    interval=5,
)

# dataset configs
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'data_dict_subset_A_train_ls.pkl',
        scene_map_file=data_root + 'data_dict_av2_train_ls_v3_scene.pkl',
        map_size=map_size,
        queue_length=1,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        split='train',
        points_num=method_para['n_points'],
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'data_dict_subset_A_val_ls.pkl',
        scene_map_file=data_root + 'data_dict_av2_val_ls_v3_scene.pkl',
        map_size=map_size,
        queue_length=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        split='val',
        points_num=method_para['n_points'],
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'data_dict_subset_A_val_ls.pkl',
        scene_map_file=data_root + 'data_dict_av2_val_ls_v3_scene.pkl',
        map_size=map_size,
        queue_length=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        split='val',
        points_num=method_para['n_points'],
        test_mode=True),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=5e-4 * (batch_size / 4),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy & schedule
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=3e-3)

evaluation = dict(interval=total_iters//6)
find_unused_parameters = True #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=total_iters//6)

runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

SyncBN = True