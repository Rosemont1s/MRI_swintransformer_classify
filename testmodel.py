from layers.build import MultiInputSwinTransformerForClassification
import torch

# 创建多输入模型实例
model = MultiInputSwinTransformerForClassification(
    img_size=(240, 240, 155),  # 修改为实际MRI尺寸
    num_classes=2,
    in_channels=1,  # 每个模态的输入通道数
    out_channels=768,
    num_modalities=5,  # 4个MRI + 1个掩码
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    fusion_method="concat",  # 可选: "concat", "add", "attention"
)

# 创建模拟输入数据
# 4个MRI模态 + 1个分割掩码
x1 = torch.randn(1, 1, 240, 240, 155)  # MRI模态1
x2 = torch.randn(1, 1, 240, 240, 155)  # MRI模态2
x3 = torch.randn(1, 1, 240, 240, 155)  # MRI模态3
x4 = torch.randn(1, 1, 240, 240, 155)  # MRI模态4
mask = torch.randint(0, 4, (1, 1, 240, 240, 155)).float()  # 分割掩码

# 将所有输入组合成列表传递给模型
inputs = [x1, x2, x3, x4, mask]
out1, out2 = model(inputs)

print("输出1形状:", out1.shape)
print("输出2形状:", out2.shape)
print("输出1:", out1)
print("输出2:", out2)

# 测试不同的融合方法
print("\n测试不同的融合方法:")

# 加法融合
model_add = MultiInputSwinTransformerForClassification(
    img_size=(240, 240, 155),  # 修改为实际MRI尺寸
    num_classes=2,
    in_channels=1,
    out_channels=768,
    feature_size=48,
    fusion_method="add",
)
out_add1, out_add2 = model_add(inputs)
print("加法融合输出形状:", out_add1.shape)

# 注意力融合
model_attn = MultiInputSwinTransformerForClassification(
    img_size=(240, 240, 155),  # 修改为实际MRI尺寸
    num_classes=2,
    in_channels=1,
    out_channels=768,
    feature_size=48,
    fusion_method="attention",
)
out_attn1, out_attn2 = model_attn(inputs)
print("注意力融合输出形状:", out_attn1.shape)