from __future__ import annotations
from collections.abc import Sequence
from .swin3d_layer import SwinTransformer, PatchMerging, PatchMergingV2
import torch
import torch.nn as nn
from typing_extensions import Final
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

class MultiInputSwinTransformerForClassification(nn.Module):
    """
    Modified from MONAI: https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/swin_unetr.py#L901
    Add classifier head and support for multiple input modalities (4 MRI + 1 mask)
    """
    patch_size: Final[int] = 2

    def __init__(
            self,
            img_size: Sequence[int] | int,
            num_classes: int,
            in_channels: int,  # 每个模态的输入通道数
            out_channels: int,
            num_modalities: int = 5,  # 默认为4个MRI + 1个mask
            depths: Sequence[int] = (2, 2, 2, 2),
            num_heads: Sequence[int] = (3, 6, 12, 24),
            feature_size: int = 24,
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            dropout_path_rate: float = 0.0,
            classifier_drop_rate: float = 0.3, 
            normalize: bool = True,
            use_checkpoint: bool = False,
            spatial_dims: int = 3,
            downsample="merging",
            fusion_method: str = "concat",  # 特征融合方法: concat, add, attention
    ) -> None:
        """
        Args:
            img_size: spatial dimension of input image.
            in_channels: dimension of input channels for each modality.
            out_channels: dimension of output channels.
            num_modalities: number of input modalities (default: 5 for 4 MRI + 1 mask).
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling.
            fusion_method: method to fuse features from different modalities.
        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}
        
        # 为每个模态创建一个Swin Transformer
        self.swin_transformers = nn.ModuleList()
        for _ in range(num_modalities):
            self.swin_transformers.append(
                SwinTransformer(
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
            )

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1) if spatial_dims == 3 else nn.AdaptiveAvgPool2d(1)
        
        self.fusion_method = fusion_method
        
        # 特征融合层
        if fusion_method == "concat":
            self.fusion_dim = out_channels * num_modalities
            self.fusion_layer = nn.Linear(self.fusion_dim, out_channels)
        elif fusion_method == "add":
            self.fusion_dim = out_channels
        elif fusion_method == "attention":
            self.fusion_dim = out_channels
            self.attention_weights = nn.Parameter(torch.ones(num_modalities, 1) / num_modalities)
            self.softmax = nn.Softmax(dim=0)
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
            
        # 分类器
        self.classifier1 = nn.Sequential(
            nn.Dropout(p=classifier_drop_rate),
            nn.Linear(out_channels, 3),
            # nn.Softmax(dim=1) #使用nn.CrossEntropyLoss()时，不需要将输出经过softmax层
        )
        self.classifier2 = nn.Sequential(
            nn.Dropout(p=classifier_drop_rate),
            nn.Linear(out_channels, 2),
            # nn.Softmax(dim=1) #使用nn.CrossEntropyLoss()时，不需要将输出经过softmax层
        )

    def forward(self, x_list):
        """
        Args:
            x_list: 包含多个模态输入的列表 [x1, x2, x3, x4, mask]
                   每个元素的形状应为 [B, C, D, H, W]
        """
        # 处理每个模态
        features = []
        for i, x in enumerate(x_list):
            feat = self.swin_transformers[i](x)  # 获取每个模态的特征
            pooled_feat = self.global_avg_pool(feat[-1])  # 使用最后一层的特征
            flattened_feat = torch.flatten(pooled_feat, 1)
            features.append(flattened_feat)
        
        # 特征融合
        if self.fusion_method == "concat":
            # 拼接所有特征
            fused_features = torch.cat(features, dim=1)
            fused_features = self.fusion_layer(fused_features)
        elif self.fusion_method == "add":
            # 简单相加
            fused_features = sum(features)
        elif self.fusion_method == "attention":
            # 注意力加权
            attention_weights = self.softmax(self.attention_weights)
            fused_features = sum(feat * weight for feat, weight in zip(features, attention_weights))
        
        # 分类
        out1 = self.classifier1(fused_features)
        out2 = self.classifier2(fused_features)
        
        return out1, out2
