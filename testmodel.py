from layers import SwinTransformerForClassification
import torch
model = SwinTransformerForClassification(
    img_size=(64,64,64),
    num_classes = 2,
    in_channels=1,
    out_channels=768,
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
)

x = torch.randn(1,1,64,64,64)
print(model(x))