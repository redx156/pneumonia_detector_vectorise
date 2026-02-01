"""
U-Net Lung Segmentation Training Pipeline
==========================================
Trains a U-Net model with ResNet18 encoder for chest X-ray lung segmentation.

Key features:
- Uses segmentation_models_pytorch for efficient U-Net implementation
- Dice + BCE combined loss for better convergence
- Medical-safe augmentations (rotation, flip, brightness)
- Proper train/val split with reproducible results
- Saves best model based on validation Dice score

Usage:
    python train_unet.py --data_dir ./data/segmentation --epochs 25 --batch_size 8

Expected folder structure:
    data_dir/
        images/
            CHNCXR_0001_0.png
            CHNCXR_0002_0.png
            ...
        masks/
            CHNCXR_0001_0_mask.png
            CHNCXR_0002_0_mask.png
            ...
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from datetime import datetime
from PIL import Image
import cv2

# Optional: albumentations for augmentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("⚠️  albumentations not found, using basic transforms")

import segmentation_models_pytorch as smp


# ============ CONFIGURATION ============

class Config:
    """Training configuration - modify these for your setup"""
    
    # Model
    ENCODER_NAME = "resnet18"
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 3
    OUT_CLASSES = 1  # Binary segmentation (lung vs background)
    
    # Training
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 25
    BATCH_SIZE = 8
    
    # Data
    IMAGE_SIZE = 512  # Standard size for lung segmentation
    VAL_SPLIT = 0.15  # 15% for validation
    
    # Augmentation limits (medical-safe)
    ROTATION_DEGREES = 15
    BRIGHTNESS_RANGE = 0.15
    CONTRAST_RANGE = 0.15
    
    # Paths
    CHECKPOINT_DIR = "checkpoints"
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reproducibility
    SEED = 42


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============ DATASET ============

class LungSegmentationDataset(Dataset):
    """
    Dataset for lung segmentation.
    
    Pairs images with their corresponding masks.
    Handles the naming convention: image_name.png -> image_name_mask.png
    """
    
    def __init__(self, image_paths: list, mask_dir: Path, transform=None, config=None):
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.transform = transform
        self.config = config or Config()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find corresponding mask
        # Image: CHNCXR_0001_0.png -> Mask: CHNCXR_0001_0_mask.png
        img_name = img_path.stem  # e.g., "CHNCXR_0001_0"
        mask_name = f"{img_name}_mask.png"
        mask_path = self.mask_dir / mask_name
        
        # Load mask (grayscale)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Resize to target size
        image = cv2.resize(image, (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
        mask = cv2.resize(mask, (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
        
        # Ensure mask is binary (0 or 1)
        mask = (mask > 127).astype(np.float32)
        
        # Apply augmentations
        if self.transform:
            if HAS_ALBUMENTATIONS:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            else:
                # Basic transform without albumentations
                image = self.transform(image)
                mask = torch.from_numpy(mask)
        else:
            # Convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask)
        
        # Ensure mask has channel dimension
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        
        return image, mask


# ============ TRANSFORMS ============

def get_train_transforms(config: Config):
    """
    Medical-safe augmentations for training.
    
    Includes:
    - Rotation (±15°): X-rays can have positioning differences
    - Horizontal flip: Lungs are roughly symmetric
    - Brightness/contrast: Handles acquisition variations
    - Normalization: ImageNet stats for pretrained encoder
    """
    if HAS_ALBUMENTATIONS:
        return A.Compose([
            A.Rotate(limit=config.ROTATION_DEGREES, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=config.BRIGHTNESS_RANGE,
                contrast_limit=config.CONTRAST_RANGE,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        # Fallback: basic transform
        import torchvision.transforms as T
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_val_transforms(config: Config):
    """Validation transforms - no augmentation, just normalize."""
    if HAS_ALBUMENTATIONS:
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        import torchvision.transforms as T
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# ============ MODEL ============

def create_model(config: Config):
    """
    Create U-Net with ResNet18 encoder.
    
    Uses segmentation_models_pytorch for a well-tested implementation.
    ResNet18 encoder is lightweight and already cached from classification model.
    """
    model = smp.Unet(
        encoder_name=config.ENCODER_NAME,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=config.IN_CHANNELS,
        classes=config.OUT_CLASSES,
    )
    return model.to(config.DEVICE)


# ============ LOSS FUNCTIONS ============

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    
    Good for imbalanced segmentation (lungs are large regions).
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice loss.
    
    BCE provides stable gradients early in training.
    Dice focuses on overall overlap quality.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ============ METRICS ============

def calculate_dice(pred, target, threshold=0.5):
    """Calculate Dice coefficient."""
    pred = (torch.sigmoid(pred) > threshold).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    if union == 0:
        return 1.0  # Both empty = perfect match
    
    dice = (2.0 * intersection) / union
    return dice.item()


def calculate_iou(pred, target, threshold=0.5):
    """Calculate Intersection over Union (Jaccard index)."""
    pred = (torch.sigmoid(pred) > threshold).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    if union == 0:
        return 1.0
    
    iou = intersection / union
    return iou.item()


# ============ TRAINING ============

def train_one_epoch(model, train_loader, criterion, optimizer, config):
    """Train for one epoch, return average loss and metrics."""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(config.DEVICE)
        masks = masks.to(config.DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_dice += calculate_dice(outputs, masks)
        
        # Progress every 20 batches
        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = running_loss / len(train_loader)
    avg_dice = running_dice / len(train_loader)
    
    return avg_loss, avg_dice


def evaluate(model, val_loader, criterion, config):
    """Evaluate model on validation set."""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            running_dice += calculate_dice(outputs, masks)
            running_iou += calculate_iou(outputs, masks)
    
    n_batches = len(val_loader)
    
    return {
        "loss": running_loss / n_batches,
        "dice": running_dice / n_batches,
        "iou": running_iou / n_batches
    }


def save_checkpoint(model, optimizer, epoch, metrics, config, filename):
    """Save model checkpoint."""
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(exist_ok=True)
    
    filepath = checkpoint_dir / filename
    
    # Save full checkpoint for resuming training
    full_checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": {
            "encoder_name": config.ENCODER_NAME,
            "image_size": config.IMAGE_SIZE,
            "learning_rate": config.LEARNING_RATE
        }
    }
    torch.save(full_checkpoint, filepath.with_suffix('.full.pth'))
    
    # Save ONLY model weights (for production inference)
    torch.save(model.state_dict(), filepath)
    
    print(f"💾 Saved checkpoint: {filepath}")
    
    return str(filepath)


# ============ DATA LOADING ============

def create_dataloaders(data_dir: str, config: Config):
    """
    Create train and validation dataloaders.
    
    Automatically pairs images with their masks and splits into train/val.
    """
    data_path = Path(data_dir)
    images_dir = data_path / "images"
    masks_dir = data_path / "masks"
    
    # Find all images that have corresponding masks
    all_images = []
    for img_path in sorted(images_dir.glob("*.png")):
        # Check if mask exists
        mask_name = f"{img_path.stem}_mask.png"
        mask_path = masks_dir / mask_name
        if mask_path.exists():
            all_images.append(img_path)
    
    print(f"Found {len(all_images)} image-mask pairs")
    
    # Split into train/val
    set_seed(config.SEED)
    random.shuffle(all_images)
    
    val_size = int(len(all_images) * config.VAL_SPLIT)
    val_images = all_images[:val_size]
    train_images = all_images[val_size:]
    
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")
    
    # Create datasets
    train_dataset = LungSegmentationDataset(
        train_images,
        masks_dir,
        transform=get_train_transforms(config),
        config=config
    )
    val_dataset = LungSegmentationDataset(
        val_images,
        masks_dir,
        transform=get_val_transforms(config),
        config=config
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader


# ============ MAIN TRAINING LOOP ============

def train(data_dir: str, config: Config):
    """
    Main training function.
    
    Training strategy:
    1. Start with pretrained ImageNet encoder
    2. Use combined BCE + Dice loss for stable training
    3. Track Dice score as primary metric
    4. Save best model based on validation Dice
    """
    print("=" * 60)
    print("🫁 LUNG SEGMENTATION TRAINING (U-Net)")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"Encoder: {config.ENCODER_NAME}")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(config.SEED)
    
    # Create model
    print("\n📦 Creating model...")
    model = create_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Create dataloaders
    print("\n📂 Loading data...")
    train_loader, val_loader = create_dataloaders(data_dir, config)
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # Loss function
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    print("\n⚖️  Using Combined BCE + Dice Loss")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training loop
    best_dice = 0.0
    best_epoch = 0
    history = []
    
    print("\n🚀 Starting training...")
    print("-" * 60)
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n📍 Epoch {epoch}/{config.EPOCHS}")
        
        # Train
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, config
        )
        
        # Evaluate
        metrics = evaluate(model, val_loader, criterion, config)
        
        # Update scheduler
        scheduler.step(metrics['dice'])
        
        # Log metrics
        print(f"   Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"   Val Loss: {metrics['loss']:.4f}")
        print(f"   Val Dice: {metrics['dice']:.4f} ← PRIMARY METRIC")
        print(f"   Val IoU: {metrics['iou']:.4f}")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            **metrics
        })
        
        # Save best model (by Dice)
        if metrics['dice'] > best_dice:
            best_dice = metrics['dice']
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch, metrics, config,
                "lung_segmentation.pth"
            )
            print(f"   🏆 New best Dice: {best_dice:.4f}")
        
        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, epoch, metrics, config,
            "lung_seg_latest.pth"
        )
    
    # Training complete
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Dice: {best_dice:.4f} (epoch {best_epoch})")
    print(f"Best model saved to: checkpoints/lung_segmentation.pth")
    print("\n📋 To use in your backend:")
    print('   The model will be automatically loaded from checkpoints/lung_segmentation.pth')
    
    # Save training history
    history_path = Path(config.CHECKPOINT_DIR) / "unet_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n📊 Training history saved to: {history_path}")
    
    return model, history


# ============ CLI ============

def main():
    parser = argparse.ArgumentParser(
        description="Train U-Net lung segmentation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python train_unet.py --data_dir ./data/segmentation
    
    # Custom settings
    python train_unet.py --data_dir ./data/segmentation --epochs 30 --batch_size 16 --lr 0.0001

Expected folder structure:
    data_dir/
        images/
            CHNCXR_0001_0.png
            ...
        masks/
            CHNCXR_0001_0_mask.png
            ...
        """
    )
    
    parser.add_argument(
        "--data_dir", type=str, default="./data/segmentation",
        help="Path to segmentation dataset (default: ./data/segmentation)"
    )
    parser.add_argument(
        "--epochs", type=int, default=25,
        help="Number of training epochs (default: 25)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate (default: 0.0001)"
    )
    parser.add_argument(
        "--image_size", type=int, default=512,
        help="Image size for training (default: 512)"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)"
    )
    
    args = parser.parse_args()
    
    # Update config with CLI arguments
    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.IMAGE_SIZE = args.image_size
    config.CHECKPOINT_DIR = args.checkpoint_dir
    
    # Run training
    train(args.data_dir, config)


if __name__ == "__main__":
    main()
