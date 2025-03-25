import torch
import torch.nn as nn
from .swin_transformer import SwinTransformer, swin_b  # 从您提供的文件导入


class MRIParallelClassificationNetwork(nn.Module):
    def __init__(self,
                 num_classes_1=2,
                 num_classes_2=2,
                 channels=5,  # 4 MRI + 1 mask
                 hidden_dim=128,
                 layers=(2, 2, 18, 2),
                 heads=(4, 8, 16, 32),
                 window_size=7):
        super(MRIParallelClassificationNetwork, self).__init__()

        # 自定义Swin Transformer
        self.swin_transformer = SwinTransformer(
            channels=channels,
            hidden_dim=hidden_dim,
            layers=layers,
            heads=heads,
            window_size=window_size,
            num_classes=hidden_dim * 8  # 为特征提取准备
        )

        # 移除原始分类头
        self.swin_transformer.mlp_head = nn.Identity()

        # 特征处理层
        self.feature_processor = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 两个独立的分类头
        self.classifier_1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes_1)
        )

        self.classifier_2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes_2)
        )

    def forward(self, x, mask):
        # 组合MRI图像和mask
        combined_input = torch.cat([x, mask], dim=1)

        # Swin Transformer特征提取
        # 注意：需要调整输入维度以匹配Swin Transformer
        combined_input = combined_input.permute(0, 2, 3, 4, 1)
        features = self.swin_transformer(combined_input)

        # 特征处理
        processed_features = self.feature_processor(features)

        # 两个独立的分类输出
        output_1 = self.classifier_1(processed_features)
        output_2 = self.classifier_2(processed_features)

        return output_1, output_2


# 模型初始化和使用示例
def main():
    # 创建模型实例
    model = MRIParallelClassificationNetwork(
        num_classes_1=2,  # 第一个分类任务
        num_classes_2=2,  # 第二个分类任务
        channels=5,  # 4 MRI + 1 mask通道
        hidden_dim=128,  # 与Swin-B配置一致
        layers=(2, 2, 18, 2),
        heads=(4, 8, 16, 32)
    )

    # 随机生成输入数据
    mri_data = torch.randn(4, 4, 240, 240, 155)  # [batch_size, channels, depth, height, width]
    mask_data = torch.randn(4, 1, 240, 240, 155)

    # 前向传播
    output_1, output_2 = model(mri_data, mask_data)
    print("Output 1 shape:", output_1.shape)
    print("Output 2 shape:", output_2.shape)


if __name__ == "__main__":
    main()