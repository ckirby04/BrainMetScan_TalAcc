"""
Lightweight 3D U-Net for brain metastasis segmentation
Optimized for consumer GPUs with limited VRAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Double convolution block: Conv3d -> BatchNorm -> ReLU -> Conv3d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout_p)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class DownBlock(nn.Module):
    """Downsampling block: MaxPool -> ConvBlock"""
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_p)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block: TransposeConv -> Concat -> ConvBlock"""
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_p)  # in_channels because of concat

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch due to padding
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class LightweightUNet3D(nn.Module):
    """
    Lightweight 3D U-Net for brain metastasis segmentation

    Architecture designed for consumer GPUs:
    - Reduced channel dimensions
    - Only 3 levels deep
    - Dropout for regularization
    - Efficient memory usage

    Args:
        in_channels: Number of input modalities (default: 4 for t1_pre, t1_gd, flair, bravo)
        out_channels: Number of output classes (default: 1 for binary segmentation)
        base_channels: Base number of feature channels (default: 16)
        depth: Number of downsampling levels (default: 3)
        dropout_p: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        in_channels=4,
        out_channels=1,
        base_channels=16,
        depth=3,
        dropout_p=0.1
    ):
        super().__init__()

        self.depth = depth
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]

        # Initial convolution
        self.inc = ConvBlock(in_channels, channels[0], dropout_p)

        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList([
            DownBlock(channels[i], channels[i+1], dropout_p)
            for i in range(depth)
        ])

        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList([
            UpBlock(channels[i+1], channels[i], dropout_p)
            for i in range(depth-1, -1, -1)
        ])

        # Final convolution
        self.outc = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        skip_connections = [x1]

        x = x1
        for down in self.down_blocks:
            x = down(x)
            skip_connections.append(x)

        # Decoder (skip last skip connection, it's the bottleneck)
        skip_connections = skip_connections[:-1][::-1]

        for i, up in enumerate(self.up_blocks):
            x = up(x, skip_connections[i])

        # Output
        x = self.outc(x)
        return x


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    Handles class imbalance better than BCE for medical imaging
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined Dice + BCE Loss
    """
    def __init__(self, dice_weight=0.7, bce_weight=0.3):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("Testing LightweightUNet3D...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=16,
        depth=3
    ).to(device)

    print(f"\nModel architecture:")
    print(model)

    print(f"\nTotal parameters: {count_parameters(model):,}")

    # Test forward pass with dummy data
    batch_size = 2
    patch_size = (96, 96, 96)
    x = torch.randn(batch_size, 4, *patch_size).to(device)

    print(f"\nInput shape: {x.shape}")

    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Test loss
    target = torch.randint(0, 2, (batch_size, 1, *patch_size)).float().to(device)
    loss_fn = CombinedLoss()
    loss = loss_fn(output, target)
    print(f"\nLoss: {loss.item():.4f}")

    print("\n✓ Model test passed!")
