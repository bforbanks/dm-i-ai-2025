import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import random
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import TumorSegmentationModel, GlobalParameterUNet
from utils import dice_score, validate_segmentation

class TumorDataset(Dataset):
    """Dataset for tumor segmentation with MIP-PET images"""
    
    def __init__(self, image_paths: List[str], mask_paths: List[str], 
                 control_paths: List[str] = None, transform=None, is_training=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.control_paths = control_paths or []
        self.transform = transform
        self.is_training = is_training
        
        # Combine patient and control data
        self.all_images = image_paths + self.control_paths
        self.all_masks = mask_paths + [None] * len(self.control_paths)  # Controls have no tumors
        
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.all_images[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.all_masks[idx]
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)  # Binary mask
        else:
            # Control image - no tumor
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Apply transformations first
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Convert mask to float and normalize
        mask = mask.float() / 255.0 if torch.is_tensor(mask) else mask.astype(np.float32) / 255.0
        
        # Ensure mask has correct shape [C, H, W] for PyTorch
        if torch.is_tensor(mask):
            if len(mask.shape) == 2:  # [H, W]
                mask = mask.unsqueeze(0)  # [1, H, W]
            elif len(mask.shape) == 3 and mask.shape[-1] == 1:  # [H, W, 1]
                mask = mask.permute(2, 0, 1)  # [1, H, W]
        else:
            if len(mask.shape) == 2:  # [H, W]
                mask = mask[np.newaxis, ...]  # [1, H, W]
            elif len(mask.shape) == 3 and mask.shape[-1] == 1:  # [H, W, 1]
                mask = mask.transpose(2, 0, 1)  # [1, H, W]
        
        # Extract global parameters for training
        global_info = self._extract_global_info(img_path, mask_path)
        
        return {
            'image': image,
            'mask': mask,
            'global_info': global_info,
            'path': img_path
        }
    
    def _extract_global_info(self, img_path: str, mask_path: str) -> Dict:
        """Extract global information for supervision"""
        # Anatomical region (simple heuristic based on filename or image analysis)
        regions = ['head_neck', 'chest', 'abdomen', 'pelvis', 'extremities', 'spine', 'other']
        
        # Simple region classification based on path or random for now
        # In practice, you'd want more sophisticated region detection
        region_id = hash(img_path) % len(regions)
        
        # Cancer presence (1 if mask_path exists, 0 for controls)
        has_cancer = 1.0 if mask_path is not None else 0.0
        
        return {
            'region_id': region_id,
            'has_cancer': has_cancer
        }

def get_transforms(is_training=True):
    """Get data augmentation transforms"""
    if is_training:
        return A.Compose([
            A.Resize(512, 384),  # Resize to manageable size
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(512, 384),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

class CombinedLoss(nn.Module):
    """Combined loss function for segmentation and global parameters"""
    
    def __init__(self, dice_weight=0.6, bce_weight=0.3, global_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.global_weight = global_weight
        
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def dice_loss(self, pred, target, smooth=1e-6):
        """Dice loss for segmentation"""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice
    
    def forward(self, predictions, targets):
        # Segmentation losses
        seg_pred = predictions['segmentation']
        seg_target = targets['mask']
        
        # Ensure target shape matches prediction shape
        if seg_pred.shape != seg_target.shape:
            # If target has extra dimensions, squeeze them
            while len(seg_target.shape) > len(seg_pred.shape):
                seg_target = seg_target.squeeze(-1)
            # If target needs channel dimension, add it
            if len(seg_target.shape) == 3 and len(seg_pred.shape) == 4:
                seg_target = seg_target.unsqueeze(1)
            # If target has wrong channel dimension order, fix it
            if seg_target.shape != seg_pred.shape and len(seg_target.shape) == 4:
                if seg_target.shape[1] != seg_pred.shape[1] and seg_target.shape[-1] == seg_pred.shape[1]:
                    seg_target = seg_target.permute(0, 3, 1, 2)
        
        dice_loss = self.dice_loss(seg_pred, seg_target)
        bce_loss = self.bce_loss(seg_pred, seg_target)
        
        # Global parameter losses
        region_pred = predictions['global_params']['region_probs']
        region_target = targets['global_info']['region_id']
        
        region_loss = self.ce_loss(region_pred, region_target.long())
        
        # Combine losses
        total_loss = (
            self.dice_weight * dice_loss +
            self.bce_weight * bce_loss +
            self.global_weight * region_loss
        )
        
        return {
            'total_loss': total_loss,
            'dice_loss': dice_loss,
            'bce_loss': bce_loss,
            'region_loss': region_loss
        }

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Prepare targets dict for loss function
        targets = {
            'mask': masks,
            'global_info': {k: v.to(device) for k, v in batch['global_info'].items()}
        }
        
        # Debug: Print shapes for first batch
        if batch_idx == 0:
            print(f"Debug - Images shape: {images.shape}")
            print(f"Debug - Target mask shape: {masks.shape}")
        
        # Forward pass
        predictions = model(images)
        
        # Debug: Print prediction shapes for first batch
        if batch_idx == 0:
            print(f"Debug - Prediction segmentation shape: {predictions['segmentation'].shape}")
        
        # Calculate loss
        loss_dict = criterion(predictions, targets)
        loss = loss_dict['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            pred_binary = (predictions['segmentation'] > 0.5).float()
            dice = dice_score(masks.cpu().numpy(), pred_binary.cpu().numpy())
            total_dice += dice
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, Dice: {dice:.4f}')
    
    return total_loss / len(dataloader), total_dice / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    total_dice = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Prepare targets dict for loss function
            targets = {
                'mask': masks,
                'global_info': {k: v.to(device) for k, v in batch['global_info'].items()}
            }
            
            # Forward pass
            predictions = model(images)
            
            # Calculate loss
            loss_dict = criterion(predictions, targets)
            loss = loss_dict['total_loss']
            
            # Calculate metrics
            pred_binary = (predictions['segmentation'] > 0.5).float()
            dice = dice_score(masks.cpu().numpy(), pred_binary.cpu().numpy())
            
            total_loss += loss.item()
            total_dice += dice
    
    return total_loss / len(dataloader), total_dice / len(dataloader)

def load_data_paths(data_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """Load paths to images and masks"""
    data_path = Path(data_dir)
    
    # Patient images and masks
    patient_img_dir = data_path / 'patients' / 'imgs'
    patient_mask_dir = data_path / 'patients' / 'labels'
    
    patient_images = sorted(list(patient_img_dir.glob('*.png')))
    patient_masks = []
    
    for img_path in patient_images:
        # Find corresponding mask
        img_name = img_path.stem
        mask_name = img_name.replace('patient_', 'segmentation_') + '.png'
        mask_path = patient_mask_dir / mask_name
        
        if mask_path.exists():
            patient_masks.append(str(mask_path))
        else:
            patient_masks.append(None)
    
    # Control images (no tumors)
    control_img_dir = data_path / 'controls' / 'imgs'
    control_images = sorted(list(control_img_dir.glob('*.png')))
    
    return (
        [str(p) for p in patient_images if patient_masks[patient_images.index(p)] is not None],
        [m for m in patient_masks if m is not None],
        [str(p) for p in control_images]
    )

def main():
    """Main training function"""
    # Configuration
    config = {
        'data_dir': 'data',
        'batch_size': 8,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints',
        'val_split': 0.2
    }
    
    print(f"Using device: {config['device']}")
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Load data paths
    patient_images, patient_masks, control_images = load_data_paths(config['data_dir'])
    
    print(f"Loaded {len(patient_images)} patient images, {len(control_images)} control images")
    
    # Split patient data into train and validation
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        patient_images, patient_masks, 
        test_size=config['val_split'], 
        random_state=42
    )
    
    # Split control images
    train_controls, val_controls = train_test_split(
        control_images, 
        test_size=config['val_split'], 
        random_state=42
    )
    
    # Create datasets
    train_dataset = TumorDataset(
        train_imgs, train_masks, train_controls,
        transform=get_transforms(is_training=True)
    )
    
    val_dataset = TumorDataset(
        val_imgs, val_masks, val_controls,
        transform=get_transforms(is_training=False)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4
    )
    
    # Initialize model, loss, and optimizer
    model = GlobalParameterUNet(in_channels=3, num_classes=1).to(config['device'])
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training loop
    best_dice = 0
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    
    for epoch in range(config['num_epochs']):
        print(f'\nEpoch {epoch+1}/{config["num_epochs"]}')
        print('-' * 60)
        
        # Train
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, config['device'])
        
        # Validate
        val_loss, val_dice = validate_epoch(model, val_loader, criterion, config['device'])
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        
        print(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best_model.pth'))
            print(f'New best model saved with Dice: {best_dice:.4f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_dices': train_dices,
                'val_dices': val_dices
            }, os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_dices, label='Train Dice')
    plt.plot(val_dices, label='Val Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.title('Training and Validation Dice Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], 'training_curves.png'))
    plt.show()
    
    print(f'\nTraining completed! Best validation Dice: {best_dice:.4f}')

if __name__ == '__main__':
    main() 