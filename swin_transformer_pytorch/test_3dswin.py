from swin_3d import swin_3d_b
import torch
if __name__ == "__main__":
    model = swin_3d_b(channels=1,  # Grayscale medical image
                      num_classes=2,  # Binary classification (e.g., tumor/no tumor)
                      window_size=5)  # Recommended window size

    # Create a sample 3D volume (batch, channel, height, width, depth)
    x = torch.randn(1, 1, 224, 224, 160)
    output = model(x)