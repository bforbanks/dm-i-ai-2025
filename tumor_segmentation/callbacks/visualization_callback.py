import torch
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2
from pathlib import Path
import wandb
from utils import dice_score


class TumorVisualizationCallback(pl.Callback):
    """
    Callback to visualize tumor segmentation results every N epochs.
    Shows the same sample image, prediction, and ground truth for consistency.
    """
    
    def __init__(self, sample_image_path=None, log_every_n_epochs=10, 
                 tumor_image_name=None, control_image_name=None,
                 additional_tumor_names=None, additional_control_names=None):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.sample_image_path = sample_image_path
        self.sample_image = None
        self.sample_mask = None
        self.control_image = None
        self.control_mask = None
        
        # Store the specified image names for selection
        self.tumor_image_name = tumor_image_name
        self.control_image_name = control_image_name
        
        # Store additional image names for compact visualization
        self.additional_tumor_names = additional_tumor_names or []
        self.additional_control_names = additional_control_names or []
        
        # Store additional images and masks
        self.additional_tumor_images = []
        self.additional_tumor_masks = []
        self.additional_control_images = []
        self.additional_control_masks = []
        
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
        """Load specific tumor and control images directly from the data directories"""
        try:
            print(f"Loading specific images: tumor='{self.tumor_image_name}', control='{self.control_image_name}'")
            print(f"Additional tumors: {self.additional_tumor_names}")
            print(f"Additional controls: {self.additional_control_names}")
            
            # Get data directory from the trainer's datamodule
            datamodule = trainer.datamodule
            data_dir = Path(datamodule.data_dir)
            
            # Load main tumor image and mask
            if self.tumor_image_name:
                self.sample_image, self.sample_mask = self._load_patient_image(data_dir, self.tumor_image_name)
                print(f"✅ Loaded main tumor image '{self.tumor_image_name}'")
                print(f"Tumor image shape: {self.sample_image.shape}")
                print(f"Tumor mask shape: {self.sample_mask.shape}")
            
            # Load main control image and mask
            if self.control_image_name:
                self.control_image, self.control_mask = self._load_control_image(data_dir, self.control_image_name)
                print(f"✅ Loaded main control image '{self.control_image_name}'")
                print(f"Control image shape: {self.control_image.shape}")
                print(f"Control mask shape: {self.control_mask.shape}")
            
            # Load additional tumor images
            for tumor_name in self.additional_tumor_names:
                try:
                    image, mask = self._load_patient_image(data_dir, tumor_name)
                    self.additional_tumor_images.append(image)
                    self.additional_tumor_masks.append(mask)
                    print(f"✅ Loaded additional tumor image '{tumor_name}'")
                except Exception as e:
                    print(f"❌ Could not load additional tumor image '{tumor_name}': {e}")
            
            # Load additional control images
            for control_name in self.additional_control_names:
                try:
                    image, mask = self._load_control_image(data_dir, control_name)
                    self.additional_control_images.append(image)
                    self.additional_control_masks.append(mask)
                    print(f"✅ Loaded additional control image '{control_name}'")
                except Exception as e:
                    print(f"❌ Could not load additional control image '{control_name}': {e}")
            
            print(f"Loaded {len(self.additional_tumor_images)}/{len(self.additional_tumor_names)} additional tumors")
            print(f"Loaded {len(self.additional_control_images)}/{len(self.additional_control_names)} additional controls")
            
        except Exception as e:
            print(f"Could not load sample images: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_patient_image(self, data_dir, patient_name):
        """Load a patient image and its corresponding mask"""
        # Patient image path
        img_path = data_dir / "patients" / "imgs" / f"{patient_name}.png"
        
        # Patient mask path (convert patient_XXX to segmentation_XXX)
        mask_name = patient_name.replace("patient_", "segmentation_")
        mask_path = data_dir / "patients" / "labels" / f"{mask_name}.png"
        
        if not img_path.exists():
            raise FileNotFoundError(f"Patient image not found: {img_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Patient mask not found: {mask_path}")
        
        # Load and preprocess image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Load and preprocess mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 0).astype(np.float32)  # Binary mask
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        return image_tensor, mask_tensor
    
    def _load_control_image(self, data_dir, control_name):
        """Load a control image (no mask needed)"""
        # Control image path
        img_path = data_dir / "controls" / "imgs" / f"{control_name}.png"
        
        if not img_path.exists():
            raise FileNotFoundError(f"Control image not found: {img_path}")
        
        # Load and preprocess image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Create empty mask for control (no tumors)
        mask_tensor = torch.zeros(1, 1, 256, 256)
        
        return image_tensor, mask_tensor
    
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
            if (self.sample_image is not None and self.sample_mask is not None and 
                self.control_image is not None and self.control_mask is not None):
                print("Sample images found, creating visualization...")
                self._log_visualization(trainer, pl_module)
            else:
                print("Warning: Sample images not available for visualization")
                print(f"  Tumor image: {self.sample_image is not None}")
                print(f"  Tumor mask: {self.sample_mask is not None}")
                print(f"  Control image: {self.control_image is not None}")
                print(f"  Control mask: {self.control_mask is not None}")
        else:
            print(f"Not creating visualization (epoch {trainer.current_epoch + 1} % {self.log_every_n_epochs} != 0)")
    
    def _log_visualization(self, trainer, pl_module):
        """Create and log visualization with multiple tumor and control images"""
        try:
            # Move samples to same device as model
            device = next(pl_module.parameters()).device
            
            # Process main tumor and control images (with safety checks)
            if self.sample_image is None or self.sample_mask is None:
                print("Warning: Main tumor image or mask is None, skipping visualization")
                return
            if self.control_image is None or self.control_mask is None:
                print("Warning: Main control image or mask is None, skipping visualization")
                return
                
            tumor_image = self.sample_image.to(device)
            tumor_mask = self.sample_mask.to(device)
            control_image = self.control_image.to(device)
            control_mask = self.control_mask.to(device)
            
            # Process additional images (only if they exist)
            additional_tumor_images = []
            additional_tumor_masks = []
            for img, mask in zip(self.additional_tumor_images, self.additional_tumor_masks):
                if img is not None and mask is not None:
                    additional_tumor_images.append(img.to(device))
                    additional_tumor_masks.append(mask.to(device))
            
            additional_control_images = []
            additional_control_masks = []
            for img, mask in zip(self.additional_control_images, self.additional_control_masks):
                if img is not None and mask is not None:
                    additional_control_images.append(img.to(device))
                    additional_control_masks.append(mask.to(device))
            
            # Get predictions for all images
            pl_module.eval()
            with torch.no_grad():
                tumor_prediction = pl_module(tumor_image)
                control_prediction = pl_module(control_image)
                
                # Get predictions for additional images
                additional_tumor_predictions = [pl_module(img) for img in additional_tumor_images]
                additional_control_predictions = [pl_module(img) for img in additional_control_images]
            pl_module.train()
            
            # Convert main tumor data to numpy for visualization
            tumor_image_np = tumor_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            tumor_mask_np = tumor_mask.squeeze().cpu().numpy()
            tumor_pred_np = tumor_prediction.squeeze().cpu().numpy()
            tumor_pred_binary = (tumor_pred_np > 0.5).astype(np.float32)
            
            # Convert main control data to numpy for visualization
            control_image_np = control_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            control_mask_np = control_mask.squeeze().cpu().numpy()
            control_pred_np = control_prediction.squeeze().cpu().numpy()
            control_pred_binary = (control_pred_np > 0.5).astype(np.float32)
            
            # Convert additional data to numpy
            additional_tumor_data = []
            for i, (img, mask, pred) in enumerate(zip(additional_tumor_images, additional_tumor_masks, additional_tumor_predictions)):
                mask_np = mask.squeeze().cpu().numpy()
                pred_np = pred.squeeze().cpu().numpy()
                pred_binary = (pred_np > 0.5).astype(np.float32)
                additional_tumor_data.append((mask_np, pred_binary))
            
            additional_control_data = []
            for i, (img, mask, pred) in enumerate(zip(additional_control_images, additional_control_masks, additional_control_predictions)):
                mask_np = mask.squeeze().cpu().numpy()
                pred_np = pred.squeeze().cpu().numpy()
                pred_binary = (pred_np > 0.5).astype(np.float32)
                additional_control_data.append((mask_np, pred_binary))
            
            # Calculate total columns needed (4 for main + additional images)
            total_tumor_cols = 4 + len(additional_tumor_data)
            total_control_cols = 4 + len(additional_control_data)
            max_cols = max(total_tumor_cols, total_control_cols)
            
            # Create visualization with dynamic columns
            fig, axes = plt.subplots(2, max_cols, figsize=(max_cols * 3, 8))
            
            # Row 1: Tumor Images
            # Main tumor image - full 4-column view
            if tumor_image_np.shape[-1] == 3:  # If RGB, convert to grayscale
                tumor_image_gray = np.mean(tumor_image_np, axis=2)
            else:
                tumor_image_gray = tumor_image_np.squeeze()
            
            axes[0, 0].imshow(tumor_image_gray, cmap='gray')
            axes[0, 0].set_title('Tumor - PET MIP')
            axes[0, 0].axis('off')
            
            # Ground truth mask
            axes[0, 1].imshow(tumor_mask_np, cmap='gray', vmin=0, vmax=1)
            axes[0, 1].set_title('Tumor - True Segmentation')
            axes[0, 1].axis('off')
            
            # Predicted segmentation
            axes[0, 2].imshow(tumor_pred_binary, cmap='gray', vmin=0, vmax=1)
            axes[0, 2].set_title('Tumor - Predicted Segmentation')
            axes[0, 2].axis('off')
            
            # Overlay showing TP, FP, FN (like utils.py)
            TP = ((tumor_pred_binary > 0) & (tumor_mask_np > 0))
            FP = ((tumor_pred_binary > 0) & (tumor_mask_np == 0))
            FN = ((tumor_pred_binary == 0) & (tumor_mask_np > 0))
            
            # Create RGB overlay: FP=red, TP=green, FN=blue
            overlay = np.zeros((*tumor_pred_binary.shape, 3), dtype=np.uint8)
            overlay[TP] = [0, 255, 0]  # Green for TP
            overlay[FP] = [255, 0, 0]  # Red for FP
            overlay[FN] = [0, 0, 255]  # Blue for FN
            
            tumor_dice = dice_score(tumor_mask_np, tumor_pred_binary)
            axes[0, 3].imshow(overlay)
            axes[0, 3].set_title(f'Tumor - Dice = {tumor_dice:.3f}')
            axes[0, 3].axis('off')
            
            # Add legend for tumor row
            green_patch = mpatches.Patch(color='green', label='TP')
            red_patch = mpatches.Patch(color='red', label='FP')
            blue_patch = mpatches.Patch(color='blue', label='FN')
            axes[0, 3].legend(handles=[green_patch, red_patch, blue_patch], loc='lower right', fontsize=8)
            
            # Additional tumor images - only colored overlays
            for i, (mask_np, pred_binary) in enumerate(additional_tumor_data):
                col_idx = 4 + i
                if col_idx < max_cols:
                    # Create overlay for additional tumor
                    TP = ((pred_binary > 0) & (mask_np > 0))
                    FP = ((pred_binary > 0) & (mask_np == 0))
                    FN = ((pred_binary == 0) & (mask_np > 0))
                    
                    overlay = np.zeros((*pred_binary.shape, 3), dtype=np.uint8)
                    overlay[TP] = [0, 255, 0]  # Green for TP
                    overlay[FP] = [255, 0, 0]  # Red for FP
                    overlay[FN] = [0, 0, 255]  # Blue for FN
                    
                    dice = dice_score(mask_np, pred_binary)
                    axes[0, col_idx].imshow(overlay)
                    axes[0, col_idx].set_title(f'Tumor {i+2} - Dice = {dice:.3f}')
                    axes[0, col_idx].axis('off')
            
            # Row 2: Control Images
            # Main control image - full 4-column view
            if control_image_np.shape[-1] == 3:  # If RGB, convert to grayscale
                control_image_gray = np.mean(control_image_np, axis=2)
            else:
                control_image_gray = control_image_np.squeeze()
            
            axes[1, 0].imshow(control_image_gray, cmap='gray')
            axes[1, 0].set_title('Control - PET MIP')
            axes[1, 0].axis('off')
            
            # Control ground truth mask
            axes[1, 1].imshow(control_mask_np, cmap='gray', vmin=0, vmax=1)
            axes[1, 1].set_title('Control - True Segmentation')
            axes[1, 1].axis('off')
            
            # Control predicted segmentation
            axes[1, 2].imshow(control_pred_binary, cmap='gray', vmin=0, vmax=1)
            axes[1, 2].set_title('Control - Predicted Segmentation')
            axes[1, 2].axis('off')
            
            # Control overlay showing TP, FP, FN
            control_TP = ((control_pred_binary > 0) & (control_mask_np > 0))
            control_FP = ((control_pred_binary > 0) & (control_mask_np == 0))
            control_FN = ((control_pred_binary == 0) & (control_mask_np > 0))
            
            # Create RGB overlay for control
            control_overlay = np.zeros((*control_pred_binary.shape, 3), dtype=np.uint8)
            control_overlay[control_TP] = [0, 255, 0]  # Green for TP
            control_overlay[control_FP] = [255, 0, 0]  # Red for FP
            control_overlay[control_FN] = [0, 0, 255]  # Blue for FN
            
            control_dice = dice_score(control_mask_np, control_pred_binary)
            axes[1, 3].imshow(control_overlay)
            axes[1, 3].set_title(f'Control - Dice = {control_dice:.3f}')
            axes[1, 3].axis('off')
            
            # Add legend for control row
            axes[1, 3].legend(handles=[green_patch, red_patch, blue_patch], loc='lower right', fontsize=8)
            
            # Additional control images - only colored overlays
            for i, (mask_np, pred_binary) in enumerate(additional_control_data):
                col_idx = 4 + i
                if col_idx < max_cols:
                    # Create overlay for additional control
                    TP = ((pred_binary > 0) & (mask_np > 0))
                    FP = ((pred_binary > 0) & (mask_np == 0))
                    FN = ((pred_binary == 0) & (mask_np > 0))
                    
                    overlay = np.zeros((*pred_binary.shape, 3), dtype=np.uint8)
                    overlay[TP] = [0, 255, 0]  # Green for TP
                    overlay[FP] = [255, 0, 0]  # Red for FP
                    overlay[FN] = [0, 0, 255]  # Blue for FN
                    
                    dice = dice_score(mask_np, pred_binary)
                    axes[1, col_idx].imshow(overlay)
                    axes[1, col_idx].set_title(f'Control {i+2} - Dice = {dice:.3f}')
                    axes[1, col_idx].axis('off')
            
            # Hide unused subplots
            for row in range(2):
                for col in range(max_cols):
                    if (row == 0 and col >= total_tumor_cols) or (row == 1 and col >= total_control_cols):
                        axes[row, col].set_visible(False)
            
            plt.tight_layout(h_pad=2, w_pad=0, pad=1.5)
            
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