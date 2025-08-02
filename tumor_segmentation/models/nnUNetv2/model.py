import torch
import torch.nn as nn
import torch.nn.functional as F
from tumor_segmentation.models.base_model import BaseModel
from typing import Optional, Tuple, Union
import os


class ConvBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.activation1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.activation2 = nn.GELU()
        
        # Residual connection if input and output channels match
        self.use_residual = (in_channels == out_channels and stride == 1)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.use_residual:
            out = out + identity
            
        out = self.activation2(out)
        return out


class EncoderBlock(nn.Module):
    """Encoder block with improved feature extraction"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        features = self.conv_block(x)
        return features, self.pool(features)


class DecoderBlock(nn.Module):
    """Decoder block with improved upsampling and feature fusion"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Use bilinear upsampling + conv instead of transposed conv for better stability
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        )
        self.conv_block = ConvBlock(in_channels, out_channels)
        
    def forward(self, x, skip_features):
        x = self.up(x)
        x = torch.cat([x, skip_features], dim=1)
        x = self.conv_block(x)
        return x


class AttentionGate(nn.Module):
    """Attention gate for better feature selection"""
    
    def __init__(self, in_channels: int, gate_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, gate_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(gate_channels, gate_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(gate_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, g):
        # x: skip connection features, g: gating signal
        theta = self.conv1(x)
        
        # Resize gating signal to match skip features spatial dimensions
        if g.shape[2:] != x.shape[2:]:
            g_resized = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        else:
            g_resized = g
            
        phi = self.conv2(g_resized)
        f = self.relu(theta + phi)
        psi = self.conv3(f)
        attention = self.sigmoid(psi)
        return x * attention


class NoAttentionGate(nn.Module):
    """Placeholder module that returns the first argument unchanged (for when attention is disabled)"""
    
    def forward(self, x, g):
        # Simply return the skip features unchanged
        return x


class nnUNetv2(BaseModel):
    """
    Improved nnU-Net v2 (2-D) implementation with better learning dynamics.
    
    Key improvements:
    - GroupNorm instead of InstanceNorm (more stable)
    - GELU activation instead of LeakyReLU (better gradients)
    - Residual connections for better gradient flow
    - Attention gates for better feature selection
    - Bilinear upsampling instead of transposed conv
    - Better weight initialization
    """
    
    def __init__(
        self, 
        in_channels: int = 3, 
        num_classes: int = 1, 
        base_channels: int = 32,
        depth: int = 5,
        lr: float = 1e-3, 
        weight_decay: float = 1e-5,
        use_pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        use_attention: bool = True
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.depth = depth
        self.use_pretrained = use_pretrained
        self.pretrained_path = pretrained_path
        self.use_attention = use_attention
        
        # Calculate channel sizes for each depth level
        self.channels = [base_channels * (2 ** i) for i in range(depth + 1)]
        
        # Initial convolution
        self.initial_conv = ConvBlock(in_channels, base_channels)
        
        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        for i in range(depth):
            in_ch = base_channels if i == 0 else self.channels[i - 1]
            self.encoder_blocks.append(EncoderBlock(in_ch, self.channels[i]))
        
        # Bottleneck
        self.bottleneck = ConvBlock(self.channels[depth - 1], self.channels[depth])
        
        # Decoder path
        self.decoder_blocks = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            in_ch = self.channels[i + 1]
            self.decoder_blocks.append(DecoderBlock(in_ch, self.channels[i]))
            
            if use_attention and i > 0:  # No attention gate for the last decoder block
                self.attention_gates.append(AttentionGate(self.channels[i], self.channels[i + 1]))
            else:
                # Add a placeholder module that handles the two-argument call
                self.attention_gates.append(NoAttentionGate())
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.channels[0], base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, num_classes, kernel_size=1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Load pre-trained weights if requested
        if use_pretrained:
            self._load_pretrained_weights()
    
    def _initialize_weights(self):
        """Initialize weights using improved initialization strategy"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use orthogonal initialization for better gradient flow
                nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _load_pretrained_weights(self):
        """
        Load pre-trained weights from TotalSegmentator v2.
        This method attempts to load weights from a specified path or downloads them.
        """
        try:
            print("=" * 60)
            print("🔍 TOTALSEGMENTATOR V2 PRE-TRAINED WEIGHT LOADING")
            print("=" * 60)
            
            # Check if pretrained path is provided
            if self.pretrained_path and os.path.exists(self.pretrained_path):
                checkpoint = torch.load(self.pretrained_path, map_location='cpu')
                print(f"✅ Loading weights from specified path: {self.pretrained_path}")
            else:
                # Try to find weights in common locations
                possible_paths = [
                    "models/nnUNetv2/pretrained/totalsegmentator_v2.pth",
                    "models/nnUNetv2/pretrained/totalsegmentator_v2.4.pth",
                    "checkpoints/totalsegmentator_v2.pth",
                    "pretrained/totalsegmentator_v2.pth"
                ]
                
                checkpoint = None
                for path in possible_paths:
                    if os.path.exists(path):
                        checkpoint = torch.load(path, map_location='cpu')
                        print(f"✅ Found weights at: {path}")
                        break
                
                if checkpoint is None:
                    print("❌ No pre-trained weights found in common locations:")
                    for path in possible_paths:
                        print(f"   - {path}: {'✅ EXISTS' if os.path.exists(path) else '❌ NOT FOUND'}")
                    print("\n📋 To use pre-trained weights:")
                    print("1. Run: python download_weights.py")
                    print("2. Or download TotalSegmentator v2 weights manually")
                    print("3. Place them in: models/nnUNetv2/pretrained/")
                    print("4. Or set pretrained_path parameter to the weights file location")
                    print("5. Continuing with random initialization...")
                    print("=" * 60)
                    return

            # Load state dict with strict=False to handle architecture differences
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"📦 Found 'model_state_dict' in checkpoint")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"📦 Found 'state_dict' in checkpoint")
            else:
                state_dict = checkpoint
                print(f"📦 Using checkpoint directly as state dict")
            
            print(f"📊 Checkpoint contains {len(state_dict)} layers")
            
            # Filter and load compatible weights
            model_state_dict = self.state_dict()
            compatible_weights = {}
            incompatible_weights = []
            
            print(f"\n🔍 ANALYZING WEIGHT COMPATIBILITY:")
            print(f"   Model has {len(model_state_dict)} layers")
            print(f"   Checkpoint has {len(state_dict)} layers")
            
            for name, param in state_dict.items():
                if name in model_state_dict and param.shape == model_state_dict[name].shape:
                    compatible_weights[name] = param
                    print(f"   ✅ {name}: {param.shape} - COMPATIBLE")
                else:
                    # Try to match by removing prefixes or suffixes
                    clean_name = name.replace('module.', '').replace('model.', '')
                    if clean_name in model_state_dict and param.shape == model_state_dict[clean_name].shape:
                        compatible_weights[clean_name] = param
                        print(f"   ✅ {name} -> {clean_name}: {param.shape} - COMPATIBLE (renamed)")
                    else:
                        incompatible_weights.append((name, param.shape))
                        if name in model_state_dict:
                            print(f"   ❌ {name}: checkpoint {param.shape} vs model {model_state_dict[name].shape} - SHAPE MISMATCH")
                        else:
                            print(f"   ❌ {name}: {param.shape} - NOT FOUND IN MODEL")
            
            print(f"\n📈 LOADING SUMMARY:")
            print(f"   ✅ Compatible layers: {len(compatible_weights)}")
            print(f"   ❌ Incompatible layers: {len(incompatible_weights)}")
            
            if compatible_weights:
                # Calculate parameter statistics
                total_model_params = sum(p.numel() for p in self.parameters())
                loaded_params = sum(p.numel() for p in compatible_weights.values())
                random_params = total_model_params - loaded_params
                
                print(f"\n📊 PARAMETER STATISTICS:")
                print(f"   Total model parameters: {total_model_params:,}")
                print(f"   Pre-trained parameters: {loaded_params:,} ({loaded_params/total_model_params*100:.1f}%)")
                print(f"   Random parameters: {random_params:,} ({random_params/total_model_params*100:.1f}%)")
                
                # Load the compatible weights
                model_state_dict.update(compatible_weights)
                self.load_state_dict(model_state_dict)
                
                print(f"\n✅ SUCCESSFULLY LOADED {len(compatible_weights)} LAYERS!")
                print(f"   🎯 Model is now initialized with {loaded_params/total_model_params*100:.1f}% pre-trained weights")
                
                # Show which layers were loaded
                print(f"\n📋 LOADED LAYERS:")
                for name in sorted(compatible_weights.keys()):
                    param = compatible_weights[name]
                    print(f"   ✅ {name}: {param.shape} ({param.numel():,} params)")
                
            else:
                print(f"\n❌ NO COMPATIBLE WEIGHTS FOUND!")
                print(f"   Continuing with random initialization...")
            
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ ERROR LOADING PRE-TRAINED WEIGHTS: {e}")
            print("   Continuing with random initialization...")
            print("=" * 60)
    
    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        
        # Encoder path with skip connections
        skip_connections = []
        
        for encoder_block in self.encoder_blocks:
            features, x = encoder_block(x)
            skip_connections.append(features)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections and attention
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_features = skip_connections[-(i + 1)]
            
            # Apply attention gate if available (nn.Identity() will just return the input unchanged)
            skip_features = self.attention_gates[i](skip_features, x)
            
            x = decoder_block(x, skip_features)
        
        # Final output
        x = self.final_conv(x)
        
        return torch.sigmoid(x)
    
    def get_model_info(self):
        """Return model information for logging"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": "nnUNetv2_improved",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "depth": self.depth,
            "base_channels": self.base_channels,
            "channels": self.channels,
            "use_pretrained": self.use_pretrained,
            "pretrained_path": self.pretrained_path,
            "use_attention": self.use_attention
        } 