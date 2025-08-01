import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
import cv2

class SpatialAttention(nn.Module):
    """Spatial attention module to focus on tumor-relevant regions"""
    def __init__(self, in_channels: int):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention

class ChannelAttention(nn.Module):
    """Channel attention module for feature importance weighting"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class GlobalParameterExtractor(nn.Module):
    """Extract global parameters like anatomical location and symmetry information"""
    def __init__(self, input_channels: int = 3):
        super(GlobalParameterExtractor, self).__init__()
        # Anatomical region classifier
        self.region_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten(),
            nn.Linear(16 * 16 * input_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # 7 body regions: head/neck, chest, abdomen, pelvis, extremities, spine, other
        )
        
        # Symmetry analyzer
        self.symmetry_analyzer = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 32),  # Symmetry features
            nn.Tanh()
        )
        
    def forward(self, x):
        # Extract anatomical region probabilities
        region_probs = F.softmax(self.region_classifier(x), dim=1)
        
        # Extract symmetry features
        symmetry_features = self.symmetry_analyzer(x)
        
        return {
            'region_probs': region_probs,
            'symmetry_features': symmetry_features
        }

class AttentionUNetEncoder(nn.Module):
    """U-Net encoder with attention mechanisms"""
    def __init__(self, in_channels: int, out_channels: int):
        super(AttentionUNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Attention mechanisms
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention(out_channels)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Apply attention
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        
        return x

class AttentionUNetDecoder(nn.Module):
    """U-Net decoder with skip connections and attention"""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super(AttentionUNetDecoder, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Attention for skip connections
        self.skip_attention = SpatialAttention(skip_channels)
        
    def forward(self, x, skip):
        x = self.upconv(x)
        
        # Apply attention to skip connection
        skip = self.skip_attention(skip)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        return x

class GlobalParameterUNet(nn.Module):
    """U-Net with global parameter integration for tumor segmentation"""
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super(GlobalParameterUNet, self).__init__()
        
        # Global parameter extractor
        self.global_extractor = GlobalParameterExtractor(in_channels)
        
        # Encoder
        self.enc1 = AttentionUNetEncoder(in_channels, 64)
        self.enc2 = AttentionUNetEncoder(64, 128)
        self.enc3 = AttentionUNetEncoder(128, 256)
        self.enc4 = AttentionUNetEncoder(256, 512)
        
        # Bottleneck with global parameter integration
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Global parameter integration layer
        self.global_integration = nn.Sequential(
            nn.Conv2d(1024 + 32 + 7, 1024, 1),  # +32 symmetry + 7 region features
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec4 = AttentionUNetDecoder(1024, 512, 512)
        self.dec3 = AttentionUNetDecoder(512, 256, 256)
        self.dec2 = AttentionUNetDecoder(256, 128, 128)
        self.dec1 = AttentionUNetDecoder(128, 64, 64)
        
        # Final classifier
        self.final_conv = nn.Conv2d(64, num_classes, 1)
        
        # Max pooling for encoder
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # Extract global parameters
        global_params = self.global_extractor(x)
        
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Integrate global parameters
        # Expand global features to match spatial dimensions
        region_features = global_params['region_probs'].unsqueeze(-1).unsqueeze(-1)
        region_features = region_features.expand(-1, -1, bottleneck.shape[2], bottleneck.shape[3])
        
        symmetry_features = global_params['symmetry_features'].unsqueeze(-1).unsqueeze(-1)
        symmetry_features = symmetry_features.expand(-1, -1, bottleneck.shape[2], bottleneck.shape[3])
        
        # Concatenate global features with bottleneck
        bottleneck_with_global = torch.cat([bottleneck, region_features, symmetry_features], dim=1)
        bottleneck_integrated = self.global_integration(bottleneck_with_global)
        
        # Decoder path
        dec4 = self.dec4(bottleneck_integrated, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)
        
        # Final prediction
        output = self.final_conv(dec1)
        
        return {
            'segmentation': torch.sigmoid(output),
            'global_params': global_params
        }

class TumorSegmentationModel:
    """Complete tumor segmentation model with preprocessing and postprocessing"""
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = GlobalParameterUNet(in_channels=3, num_classes=1).to(device)
        
        if model_path:
            self.load_model(model_path)
    
    def preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess MIP-PET image for model input"""
        # Ensure 3 channels
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return img_tensor
    
    def postprocess_segmentation(self, pred: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        """Convert model output to final segmentation format"""
        # Convert to numpy
        pred_np = pred.cpu().numpy().squeeze()
        
        # Threshold and convert to binary
        binary_mask = (pred_np > 0.5).astype(np.uint8) * 255
        
        # Resize to original shape if needed
        if binary_mask.shape != original_shape:
            binary_mask = cv2.resize(binary_mask, (original_shape[1], original_shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
        
        # Convert to RGB format
        segmentation = np.stack([binary_mask, binary_mask, binary_mask], axis=-1)
        
        return segmentation
    
    def analyze_symmetry(self, img: np.ndarray) -> Dict[str, float]:
        """Analyze image symmetry to detect bilateral patterns"""
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
        
        h, w = img_gray.shape
        left_half = img_gray[:, :w//2]
        right_half = np.fliplr(img_gray[:, w//2:])
        
        # Resize to same size if needed
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        # Calculate symmetry score
        diff = np.abs(left_half.astype(np.float32) - right_half.astype(np.float32))
        symmetry_score = 1.0 - (np.mean(diff) / 255.0)
        
        return {
            'symmetry_score': symmetry_score,
            'asymmetric_regions': diff > 50  # Threshold for significant asymmetry
        }
    
    def predict(self, img: np.ndarray) -> np.ndarray:
        """Main prediction function"""
        original_shape = img.shape[:2]
        
        # Preprocess
        img_tensor = self.preprocess_image(img)
        
        # Model inference
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # Postprocess
        segmentation = self.postprocess_segmentation(output['segmentation'], original_shape)
        
        return segmentation
    
    def predict_with_analysis(self, img: np.ndarray) -> Dict:
        """Prediction with additional analysis"""
        original_shape = img.shape[:2]
        
        # Preprocess
        img_tensor = self.preprocess_image(img)
        
        # Model inference
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # Postprocess
        segmentation = self.postprocess_segmentation(output['segmentation'], original_shape)
        
        # Additional analysis
        symmetry_analysis = self.analyze_symmetry(img)
        
        return {
            'segmentation': segmentation,
            'global_params': output['global_params'],
            'symmetry_analysis': symmetry_analysis,
            'confidence': float(torch.max(output['segmentation']).cpu())
        }
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device)) 