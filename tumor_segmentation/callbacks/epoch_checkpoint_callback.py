import lightning.pytorch as pl
from pathlib import Path
import torch


class EpochCheckpointCallback(pl.Callback):
    """
    Custom callback to save a checkpoint at a specific epoch.
    This is useful for saving checkpoints at important milestones during training.
    """
    
    def __init__(self, target_epoch: int, dirpath: str = "checkpoints", filename: str = "epoch_specific"):
        super().__init__()
        self.target_epoch = target_epoch
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.saved = False  # Track if we've already saved for this epoch
        
        # Create directory if it doesn't exist
        self.dirpath.mkdir(parents=True, exist_ok=True)
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Save checkpoint at the target epoch."""
        current_epoch = trainer.current_epoch
        
        # Check if we've reached the target epoch and haven't saved yet
        if current_epoch == self.target_epoch and not self.saved:
            # Create the full file path
            checkpoint_path = self.dirpath / f"{self.filename}.ckpt"
            
            # Save the checkpoint
            trainer.save_checkpoint(checkpoint_path)
            
            # Mark as saved to avoid duplicate saves
            self.saved = True
            
            # Checkpoint saved silently
            
            # Log the checkpoint path for easy access
            if hasattr(trainer, 'log'):
                trainer.log("specific_checkpoint_path", str(checkpoint_path), on_epoch=True)
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Reset saved flag at the start of training."""
        self.saved = False
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log final status."""
        # Status logged silently 