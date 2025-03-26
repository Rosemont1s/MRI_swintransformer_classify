from __future__ import annotations
from collections.abc import Sequence
from .swin3d_layer import SwinTransformer, PatchMerging, PatchMergingV2
import torch
import torch.nn as nn
from typing_extensions import Final
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
class SwinTransformerForClassification(nn.Module):
    """
    Modified from MONAI: https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/swin_unetr.py#L901
    Add classifier head

    """
    patch_size: Final[int] = 2

    def __init__(
            self,
            img_size: Sequence[int] | int,
            num_classes: int,
            in_channels: int,
            out_channels: int,
            depths: Sequence[int] = (2, 2, 2, 2),
            num_heads: Sequence[int] = (3, 6, 12, 24),
            feature_size: int = 24,
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            dropout_path_rate: float = 0.0,
            normalize: bool = True,
            use_checkpoint: bool = False,
            spatial_dims: int = 3,
            downsample="merging",

    ) -> None:
        """
        Args:
            img_size: spatial dimension of input image.
                This argument is only used for checking that the input image size is divisible by the patch size.
                The tensor passed to forward() can have a dynamic shape as long as its spatial dimensions are divisible by 2**5.
                It will be removed in an upcoming version.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            use_v2: using swinunetr_v2, which adds a residual convolution block at the beggining of each swin stage.

        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}
        self.swin_transformer = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_sizes,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=False,
        )

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1) if spatial_dims == 3 else nn.AdaptiveAvgPool2d(1)

        self.classifier1 = nn.Sequential(
            nn.Linear(out_channels, num_classes),
            # nn.Softmax(dim=1) #使用nn.CrossEntropyLoss()時，不需要將輸出經過softmax層，否則計算的損失會有誤 #https://discuss.pytorch.org/t/which-is-the-right-loss/61135/9
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(out_channels, num_classes),
            # nn.Softmax(dim=1) #使用nn.CrossEntropyLoss()時，不需要將輸出經過softmax層，否則計算的損失會有誤 #https://discuss.pytorch.org/t/which-is-the-right-loss/61135/9
        )

    def forward(self, x):
        x = self.swin_transformer(x)  # ([20, 768, 2, 2, 2])
        x = self.global_avg_pool(x[-1])  # Assume the output from Swin Transformer is the last layer's output
        x = torch.flatten(x, 1)
        out1 = self.classifier1(x)
        out2 = self.classifier2(x)
        return out1, out2
