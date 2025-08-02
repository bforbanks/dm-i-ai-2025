import torch
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import wandb


class TumorVisualizationCallback(pl.Callback):
    """
    Callback to visualize tumor segmentation results every N epochs.
    Shows the same sample image, prediction, and ground truth for consistency.
    """
    
    def __init__(self, sample_image_path=None, log_every_n_epochs=10):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.sample_image_path = sample_image_path
        self.sample_image = None
        self.sample_mask = None
        
    def on_train_start(self, trainer, pl_module):
        """Load sample image and mask on training start"""
        if self.sample_image_path is None:
            # Try to find a sample tumor image automatically
            self._find_sample_image(trainer)
        
        if self.sample_image_path and Path(self.sample_image_path).exists():
            self._load_sample_data()
        else:
            print("Warning: Could not find sample image for visualization")
    
    def _find_sample_image(self, trainer):
        """Try to find a sample tumor image from the dataset"""
        try:
            print("Trying to find sample image from validation set...")
            
            # Get a batch from the validation dataloader
            val_dataloader = trainer.val_dataloaders
            batch = next(iter(val_dataloader))
            images, masks = batch
            
            print(f"Found batch with {images.shape[0]} images, shape: {images.shape}")
            print(f"Masks shape: {masks.shape}")
            
            # Find an image with a tumor (non-zero mask)
            for i in range(images.shape[0]):
                mask = masks[i]
                mask_sum = mask.sum().item()
                print(f"Image {i}: mask sum = {mask_sum}")
                
                if mask_sum > 0:  # Has tumor
                    # Save this as our sample
                    self.sample_image = images[i].unsqueeze(0)  # Add batch dimension
                    self.sample_mask = masks[i].unsqueeze(0)    # Add batch dimension
                    print(f"✅ Using sample image {i} from validation set for visualization")
                    print(f"Sample image shape: {self.sample_image.shape}")
                    print(f"Sample mask shape: {self.sample_mask.shape}")
                    return
            
            # If no tumor images found, use the first image anyway
            print("No tumor images found, using first image for visualization")
            self.sample_image = images[0].unsqueeze(0)
            self.sample_mask = masks[0].unsqueeze(0)
            print(f"Using first image for visualization")
            print(f"Sample image shape: {self.sample_image.shape}")
            print(f"Sample mask shape: {self.sample_mask.shape}")
                
        except Exception as e:
            print(f"Could not find sample image automatically: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_sample_data(self):
        """Load sample image and mask from file"""
        try:
            # Load image
            image = cv2.imread(self.sample_image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            image = cv2.resize(image, (256, 256))
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Convert to tensor and add batch dimension
            self.sample_image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            
            # Create dummy mask (you can modify this to load actual mask)
            self.sample_mask = torch.zeros(1, 1, 256, 256)
            
            print(f"Loaded sample image from {self.sample_image_path}")
            
        except Exception as e:
            print(f"Could not load sample image: {e}")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Log visualization every N epochs"""
        print(f"Visualization callback: Epoch {trainer.current_epoch + 1}, log_every_n_epochs={self.log_every_n_epochs}")
        if (trainer.current_epoch + 1) % self.log_every_n_epochs == 0:
            print(f"Should create visualization for epoch {trainer.current_epoch + 1}")
            if self.sample_image is not None:
                print("Sample image found, creating visualization...")
                self._log_visualization(trainer, pl_module)
            else:
                print("Warning: No sample image available for visualization")
        else:
            print(f"Not creating visualization (epoch {trainer.current_epoch + 1} % {self.log_every_n_epochs} != 0)")
    
    def _log_visualization(self, trainer, pl_module):
        """Create and log visualization"""
        try:
            # Move sample to same device as model
            device = next(pl_module.parameters()).device
            sample_image = self.sample_image.to(device)
            sample_mask = self.sample_mask.to(device)
            
            # Get prediction
            pl_module.eval()
            with torch.no_grad():
                prediction = pl_module(sample_image)
            pl_module.train()
            
            # Convert to numpy for visualization
            image_np = sample_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            mask_np = sample_mask.squeeze().cpu().numpy()
            pred_np = prediction.squeeze().cpu().numpy()
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(image_np)
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            # Ground truth mask
            axes[1].imshow(image_np)
            axes[1].imshow(mask_np, alpha=0.5, cmap='Reds')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Prediction
            axes[2].imshow(image_np)
            axes[2].imshow(pred_np, alpha=0.5, cmap='Blues')
            axes[2].set_title(f'Prediction (Epoch {trainer.current_epoch + 1})')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Log to wandb if available
            if trainer.logger and hasattr(trainer.logger, 'experiment'):
                trainer.logger.experiment.log({
                    f"tumor_visualization_epoch_{trainer.current_epoch + 1}": wandb.Image(fig)
                })
            
            # Save locally
            save_path = Path("visualizations")
            save_path.mkdir(exist_ok=True)
            filename = f"tumor_vis_epoch_{trainer.current_epoch + 1}.png"
            filepath = save_path / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved tumor visualization for epoch {trainer.current_epoch + 1} to {filepath.absolute()}")
            
            # Check if file was actually created
            if filepath.exists():
                print(f"✅ Visualization file created successfully: {filepath}")
            else:
                print(f"❌ Visualization file was not created: {filepath}")
            
        except Exception as e:
            print(f"Error creating visualization: {e}") 