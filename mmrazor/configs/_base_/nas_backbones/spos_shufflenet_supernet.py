from mmrazor.models import (OneShotMutableOP, SearchableShuffleNetV2,
                            ShuffleBlock, ShuffleXception)

# _STAGE_MUTABLE = dict(
#     type='mmrazor.OneShotMutableOP',
#     candidates=dict(
#         shuffle_3x3=dict(type='mmrazor.ShuffleBlock', kernel_size=3),
#         shuffle_5x5=dict(type='mmrazor.ShuffleBlock', kernel_size=5),
#         shuffle_7x7=dict(type='mmrazor.ShuffleBlock', kernel_size=7),
#         shuffle_xception=dict(type='mmrazor.ShuffleXception')))

_STAGE_MUTABLE = OneShotMutableOP(
    candidates=dict(
        shuffle_3x3=ShuffleBlock(in_channels='???', out_channels='???', kernel_size=3),
        shuffle_5x5=ShuffleBlock(in_channels='???', out_channels='???', kernel_size=5),
        shuffle_7x7=ShuffleBlock(in_channels='???', out_channels='???', kernel_size=7),
        shuffle_xception=ShuffleXception(in_channels='???', out_channels='???')
    ),
    module_kwargs='???'
)

arch_setting = [
    # Parameters to build layers. 3 parameters are needed to construct a
    # layer, from left to right: channel, num_blocks, mutable_cfg.
    [64, 4, _STAGE_MUTABLE],
    [160, 4, _STAGE_MUTABLE],
    [320, 8, _STAGE_MUTABLE],
    [640, 4, _STAGE_MUTABLE]
]

nas_backbone = SearchableShuffleNetV2(
    widen_factor=1.0,
    arch_setting=arch_setting)
