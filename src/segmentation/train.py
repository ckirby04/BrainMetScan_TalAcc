"""
Training script for brain metastasis segmentation U-Net
Optimized for consumer GPUs with mixed precision training
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json

from dataset import BrainMetDataset, get_train_val_split
from unet import LightweightUNet3D, CombinedLoss, count_parameters


def dice_coefficient(pred, target, threshold=0.5):
    """Calculate Dice coefficient"""
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-8)

    return dice.item()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0

    pbar = tqdm(loader, desc="Training")
    for batch_idx, (images, masks, _) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        dice = dice_coefficient(outputs, masks)
        total_loss += loss.item()
        total_dice += dice

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}'
        })

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)

    return avg_loss, avg_dice


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_dice = 0

    pbar = tqdm(loader, desc="Validation")
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        dice = dice_coefficient(outputs, masks)
        total_loss += loss.item()
        total_dice += dice

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}'
        })

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)

    return avg_loss, avg_dice


def train(args):
    """Main training function"""

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print("\nLoading dataset...")
    full_dataset = BrainMetDataset(
        data_dir=args.data_dir,
        sequences=['t1_pre', 't1_gd', 'flair', 'bravo'],
        patch_size=tuple(args.patch_size),
        metadata_path=args.metadata_path
    )

    # Train/val split
    train_cases, val_cases = get_train_val_split(
        args.data_dir,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # Get indices for train/val
    case_to_idx = {case.name: idx for idx, case in enumerate(full_dataset.cases)}
    train_indices = [case_to_idx[case.name] for case in train_cases]
    val_indices = [case_to_idx[case.name] for case in val_cases]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=args.base_channels,
        depth=args.depth,
        dropout_p=args.dropout
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Loss and optimizer
    criterion = CombinedLoss(
        dice_weight=args.dice_weight,
        bce_weight=args.bce_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...\n")

    best_dice = 0
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': [],
        'lr': []
    }

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # Validate
        val_loss, val_dice = validate(
            model, val_loader, criterion, device
        )

        # Update scheduler
        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['lr'].append(current_lr)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        print(f"LR: {current_lr:.6f}\n")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_loss': val_loss,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"[SAVED] Best model (Dice: {val_dice:.4f})\n")

        # Save checkpoint every N epochs
        if epoch % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_loss': val_loss,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))

    # Save training history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Models saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train brain metastasis segmentation model")

    # Data
    parser.add_argument('--data_dir', type=str, default='../../train',
                        help='Path to training data directory')
    parser.add_argument('--metadata_path', type=str, default='../../metadata.csv',
                        help='Path to metadata CSV')
    parser.add_argument('--output_dir', type=str, default='../../../models',
                        help='Output directory for models')

    # Model
    parser.add_argument('--base_channels', type=int, default=16,
                        help='Base number of channels in U-Net')
    parser.add_argument('--depth', type=int, default=3,
                        help='Depth of U-Net')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (reduce if OOM)')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[96, 96, 96],
                        help='3D patch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')

    # Loss
    parser.add_argument('--dice_weight', type=float, default=0.7,
                        help='Weight for Dice loss')
    parser.add_argument('--bce_weight', type=float, default=0.3,
                        help='Weight for BCE loss')

    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)
