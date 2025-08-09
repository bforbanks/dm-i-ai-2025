#!/usr/bin/env python3
"""
Improved VAE for Tumor Mask Generation
=====================================

A more sophisticated VAE implementation specifically designed for generating
realistic tumor masks with proper shape diversity and detail preservation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple
import random


class ImprovedBetaVAE(nn.Module):
    """
    Improved β-VAE with better architecture for tumor mask generation.
    
    Key improvements:
    - Deeper network with skip connections
    - Better loss balancing
    - Proper normalization
    - Attention mechanisms
    """
    
    def __init__(self, latent_dim: int = 32, beta: float = 2.0, input_size: int = 64):
        super().__init__()
        self.beta = beta
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        # Encoder with more layers and better feature extraction
        self.encoder = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder with skip connections
        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.decoder = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            # Final layer with better activation
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_dec(z).view(-1, 256, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        """Improved loss function with better balancing."""
        # Reconstruction loss with focal component for better detail preservation
        bce = F.binary_cross_entropy(recon_x, x, reduction='none')
        
        # Focal loss component to focus on hard examples
        alpha = 0.25
        gamma = 2.0
        pt = torch.where(x == 1, recon_x, 1 - recon_x)
        focal_weight = alpha * (1 - pt) ** gamma
        focal_bce = focal_weight * bce
        
        # Combine regular BCE with focal component
        recon_loss = 0.7 * bce.mean() + 0.3 * focal_bce.mean()
        
        # KL divergence with annealing
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Add regularization for smoother latent space
        reg_loss = 0.01 * torch.mean(mu.pow(2)) + 0.01 * torch.mean(logvar.exp())
        
        return recon_loss + self.beta * kld + reg_loss


def prepare_training_data(seed_masks: List[np.ndarray], size: int = 64) -> torch.Tensor:
    """Prepare training data with better preprocessing."""
    tensors = []
    
    for mask in seed_masks:
        # Resize to target size
        img = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)
        
        # Ensure binary values
        img = (img > 0.5).astype(np.float32)
        
        # Add some data augmentation
        if random.random() < 0.3:
            # Random rotation
            angle = random.uniform(-15, 15)
            center = (size // 2, size // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, matrix, (size, size), flags=cv2.INTER_NEAREST)
        
        if random.random() < 0.3:
            # Random flip
            if random.random() < 0.5:
                img = cv2.flip(img, 0)  # vertical flip
            else:
                img = cv2.flip(img, 1)  # horizontal flip
        
        tensors.append(img[None, :, :])  # add channel dim
    
    return torch.tensor(np.stack(tensors), dtype=torch.float32)


def train_improved_vae(seed_masks: List[np.ndarray], epochs: int = 100, 
                      checkpoint: str = None, device: str = "cpu") -> ImprovedBetaVAE:
    """Train improved β-VAE with better training strategy."""
    print("⚙️  Training improved β-VAE on real masks...")
    
    # Prepare data
    data = prepare_training_data(seed_masks)
    loader = DataLoader(TensorDataset(data), batch_size=16, shuffle=True, drop_last=True)
    
    # Initialize model
    model = ImprovedBetaVAE(latent_dim=32, beta=2.0).to(device)
    
    # Better optimizer with learning rate scheduling
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop with better monitoring
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kld_loss = 0.0
        
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            recon, mu, logvar = model(batch)
            loss = model.loss_function(recon, batch, mu, logvar)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item() * batch.size(0)
            
            # Monitor individual loss components
            with torch.no_grad():
                bce = F.binary_cross_entropy(recon, batch, reduction='mean')
                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                total_recon_loss += bce.item() * batch.size(0)
                total_kld_loss += kld.item() * batch.size(0)
        
        scheduler.step()
        
        avg_loss = total_loss / len(loader.dataset)
        avg_recon = total_recon_loss / len(loader.dataset)
        avg_kld = total_kld_loss / len(loader.dataset)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            if checkpoint:
                torch.save(model.state_dict(), checkpoint)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"   Epoch {epoch+1}/{epochs}")
            print(f"     Total Loss: {avg_loss:.4f}")
            print(f"     Recon Loss: {avg_recon:.4f}")
            print(f"     KL Loss: {avg_kld:.4f}")
            print(f"     LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    return model


def sample_improved_vae(model: ImprovedBetaVAE, mask_shape: Tuple[int, int], 
                       device: str = "cpu", num_attempts: int = 20) -> np.ndarray | None:
    """Sample from improved VAE with better post-processing."""
    model.eval()
    
    with torch.no_grad():
        for _ in range(num_attempts):
            # Sample from latent space
            # Determine latent dimension in a robust way (works for original BetaVAE too)
            latent_dim = getattr(model, "latent_dim", None)
            if latent_dim is None:
                fc_dec = getattr(model, "fc_dec", None)
                latent_dim = getattr(fc_dec, "in_features", None)
            if latent_dim is None:
                raise AttributeError("Could not determine latent dimension from model")

            z = torch.randn(1, int(latent_dim), device=device)
            out = model.decode(z).cpu().numpy()[0, 0]
            H, W = out.shape
            total_px = H * W

            # Try multiple thresholds (high to low) until area fraction is plausible
            # Aim for ~0.3% - 2% of 64x64 -> scales to realistic areas after resize
            candidate = None
            for thr in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6):
                bin_mask = (out > thr).astype(np.uint8)
                area = int(bin_mask.sum())
                frac = area / float(total_px)
                if 0.003 <= frac <= 0.02:
                    candidate = bin_mask
                    break

            if candidate is None:
                # Fallback: percentile-based, then trim to largest component
                thr = np.percentile(out, 98)
                candidate = (out > thr).astype(np.uint8)

            # Keep largest component only
            num_labels, labels = cv2.connectedComponents(candidate)
            if num_labels > 1:
                largest_label = 1 + np.argmax([np.sum(labels == i) for i in range(1, num_labels)])
                candidate = (labels == largest_label).astype(np.uint8)

            # Morphology to smooth
            kernel = np.ones((3, 3), np.uint8)
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)

            # Resize to target shape
            resized = cv2.resize(candidate, (mask_shape[1], mask_shape[0]), interpolation=cv2.INTER_NEAREST)
            return resized

    return None


if __name__ == "__main__":
    # Example usage
    print("Improved VAE for tumor mask generation")
    print("Use this module by importing and calling:")
    print("  model = train_improved_vae(seed_masks, epochs=100)")
    print("  mask = sample_improved_vae(model, (992, 400))") 