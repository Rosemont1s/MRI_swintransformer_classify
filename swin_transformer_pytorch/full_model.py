import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入 Swin Transformer
from swin_transformer import SwinTransformer3D, swin_t


class MRIClassificationNetwork(nn.Module):
    def __init__(self,
                 num_input_channels=4,
                 swin_variant='3d',
                 num_classes1=2,
                 num_classes2=2,
                 window_size=7,
                 relative_pos_embedding=True):
        super().__init__()

        # 初始卷积层，融合 MR 图像和 mask
        self.input_conv = nn.Conv3d(num_input_channels + 1, num_input_channels, kernel_size=3, padding=1)

        # 3D Swin Transformer
        self.swin = SwinTransformer3D(
            channels=num_input_channels,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding
        )

        # 移除原分类头
        self.swin.mlp_head = nn.Identity()

        # 特征维度
        swin_output_dim = self.swin.stage4.patch_partition.downsample.out_channels * 8

        # 两个独立的分类头
        self.classification1 = nn.Sequential(
            nn.LayerNorm(swin_output_dim),
            nn.Linear(swin_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes1)
        )

        self.classification2 = nn.Sequential(
            nn.LayerNorm(swin_output_dim),
            nn.Linear(swin_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes2)
        )

    def forward(self, x, mask):
        # x: [4, 240, 240, 155]
        # mask: [240, 240, 155]

        # 添加 batch 维度并融合 mask
        x = x.unsqueeze(0)  # [1, 4, 240, 240, 155]
        mask = mask.unsqueeze(0).unsqueeze(0) # [1, 1, 240, 240, 155]

        # 融合输入
        x_combined = torch.cat([x, mask], dim=1)  # [1, 5, 240, 240, 155]

        # 预处理
        x_processed = self.input_conv(x_combined)  # [1, 4, 240, 240, 155]

        # Swin Transformer
        swin_features = self.swin(x_processed)

        # 分类
        output1 = self.classification1(swin_features)
        output2 = self.classification2(swin_features)

        return output1, output2


# 测试模型
def test_model():
    model = MRIClassificationNetwork()

    # 测试输入
    x = torch.randn(4, 240, 240, 155)  # 4 MR 图像
    mask = torch.randn(240, 240, 155)  # Mask

    # 前向传播
    output1, output2 = model(x, mask)

    print("Input X shape:", x.shape)
    print("Mask shape:", mask.shape)
    print("Output 1 shape:", output1.shape)
    print("Output 2 shape:", output2.shape)


if __name__ == "__main__":
    test_model()