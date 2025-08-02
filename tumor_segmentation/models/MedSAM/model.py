import torch
import torch.nn as nn
import torch.nn.functional as F
from tumor_segmentation.models.base_model import BaseModel
import math
import os
from typing import Optional, Tuple, List


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=1024, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, n_patches, embed_dim
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()  # Simplified for now
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer"""
    def __init__(self, img_size=1024, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


class PromptEncoder(nn.Module):
    """Prompt encoder for SAM"""
    def __init__(self, embed_dim=256, image_embedding_size=(64, 64), input_image_size=(1024, 1024), mask_in_chans=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.mask_in_chans = mask_in_chans

        self.point_embeddings = nn.ModuleList([
            nn.Embedding(1, embed_dim) for i in range(1)
        ])
        
        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            nn.LayerNorm([mask_in_chans // 4, image_embedding_size[0] * 2, image_embedding_size[1] * 2]),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            nn.LayerNorm([mask_in_chans, image_embedding_size[0], image_embedding_size[1]]),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )

    def forward(self, points=None, boxes=None, masks=None):
        sparse_embeddings = None
        if points is not None:
            point_embeddings = [pe(0) for pe in self.point_embeddings]
            sparse_embeddings = torch.stack(point_embeddings, dim=1)

        if masks is not None:
            mask_embeddings = self.mask_downscaling(masks)
        else:
            mask_embeddings = None

        return sparse_embeddings, mask_embeddings


class MaskDecoder(nn.Module):
    """Simplified mask decoder for SAM"""
    def __init__(self, num_multimask_outputs=1, transformer_dim=256, vit_dim=768):
        super().__init__()
        self.num_multimask_outputs = num_multimask_outputs
        self.transformer_dim = transformer_dim
        self.vit_dim = vit_dim

        # Simple decoder network
        self.decoder = nn.Sequential(
            nn.Conv2d(transformer_dim, transformer_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(transformer_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(transformer_dim // 2, transformer_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(transformer_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(transformer_dim // 4, 1, kernel_size=1),
        )

        # Upscaling layers
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            nn.LayerNorm([transformer_dim // 4, 64, 64]),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.LayerNorm([transformer_dim // 8, 128, 128]),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 8, 1, kernel_size=1),
        )

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output):
        # Add dense prompt embeddings if provided
        if dense_prompt_embeddings is not None:
            image_embeddings = image_embeddings + dense_prompt_embeddings
        
        # Process through decoder
        masks = self.decoder(image_embeddings)
        
        # Ensure correct shape for multiple masks
        if masks.shape[1] == 1:
            masks = masks.repeat(1, self.num_multimask_outputs, 1, 1)
        
        # Dummy IoU predictions
        b = masks.shape[0]
        iou_predictions = torch.ones(b, self.num_multimask_outputs, device=masks.device) * 0.5
        
        return masks, iou_predictions


class MedSAM(BaseModel):
    """
    MedSAM model using ViT-B/16 backbone for tumor segmentation.
    Inherits from BaseModel for consistent training/validation logic.
    """

    def __init__(
        self, 
        in_channels: int = 3, 
        num_classes: int = 1, 
        lr=1e-4, 
        weight_decay=1e-5,
        img_size: int = 1024,
        vit_patch_size: int = 16,
        vit_embed_dim: int = 768,
        vit_depth: int = 12,
        vit_num_heads: int = 12,
        transformer_dim: int = 256,
        pretrained_weights_path: Optional[str] = None
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        
        self.img_size = img_size
        self.vit_patch_size = vit_patch_size
        self.vit_embed_dim = vit_embed_dim
        self.transformer_dim = transformer_dim
        
        # Vision Transformer backbone
        self.image_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=vit_patch_size,
            in_chans=in_channels,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_num_heads,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=nn.LayerNorm
        )
        
        # Prompt encoder
        self.prompt_encoder = PromptEncoder(
            embed_dim=transformer_dim,
            image_embedding_size=(img_size // vit_patch_size, img_size // vit_patch_size),
            input_image_size=(img_size, img_size),
            mask_in_chans=16
        )
        
        # Mask decoder
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=1,  # Single mask output for binary segmentation
            transformer_dim=transformer_dim,
            vit_dim=vit_embed_dim
        )
        
        # Adapter to convert ViT output to SAM format
        self.image_adapter = nn.Sequential(
            nn.Linear(vit_embed_dim, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.GELU()
        )
        
        # Final output layer for binary segmentation
        self.final_conv = nn.Conv2d(1, num_classes, 1)
        
        # Load pretrained weights if provided
        if pretrained_weights_path:
            self.load_pretrained_weights(pretrained_weights_path)
    
    def load_pretrained_weights(self, weights_path: str):
        """Load pretrained MedSAM weights"""
        try:
            # Check if file exists
            if not os.path.exists(weights_path):
                print(f"Warning: Weights file not found at {weights_path}")
                print("Training will start from scratch.")
                return
            
            print(f"Loading pretrained weights from {weights_path}")
            checkpoint = torch.load(weights_path, map_location='cpu')
            
            # Print checkpoint keys to understand structure
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Try different possible key names
            encoder_keys = ['image_encoder', 'encoder', 'backbone', 'vit']
            decoder_keys = ['mask_decoder', 'decoder', 'head']
            
            # Load image encoder weights
            encoder_loaded = False
            for key in encoder_keys:
                if key in checkpoint:
                    try:
                        self.image_encoder.load_state_dict(checkpoint[key], strict=False)
                        print(f"Loaded image encoder weights from key '{key}'")
                        encoder_loaded = True
                        break
                    except Exception as e:
                        print(f"Failed to load encoder from key '{key}': {e}")
            
            if not encoder_loaded:
                print("Warning: Could not load image encoder weights")
            
            # Load decoder weights if available
            decoder_loaded = False
            for key in decoder_keys:
                if key in checkpoint:
                    try:
                        self.mask_decoder.load_state_dict(checkpoint[key], strict=False)
                        print(f"Loaded decoder weights from key '{key}'")
                        decoder_loaded = True
                        break
                    except Exception as e:
                        print(f"Failed to load decoder from key '{key}': {e}")
            
            if not decoder_loaded:
                print("Warning: Could not load decoder weights")
                
        except Exception as e:
            print(f"Warning: Could not load pretrained weights from {weights_path}: {e}")
            print("Training will start from scratch.")

    def forward(self, x):
        # Ensure input is the right size
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        # Image encoder (ViT)
        image_embeddings = self.image_encoder(x)  # [B, num_patches + 1, embed_dim]
        
        # Remove CLS token and reshape to spatial format
        image_embeddings = image_embeddings[:, 1:, :]  # Remove CLS token
        B, num_patches, embed_dim = image_embeddings.shape
        H = W = int(math.sqrt(num_patches))
        image_embeddings = image_embeddings.transpose(1, 2).reshape(B, embed_dim, H, W)
        
        # Debug: Print intermediate shapes
        if self.training and torch.rand(1).item() < 0.01:  # Only print 1% of the time
            print(f"ViT output shape: {image_embeddings.shape}")
        
        # Adapt ViT embeddings to SAM format
        # image_embeddings shape: [B, embed_dim, H, W] -> [B, H*W, embed_dim]
        B, embed_dim, H, W = image_embeddings.shape
        # Reshape to [B*H*W, embed_dim] for linear layer
        image_embeddings_flat = image_embeddings.permute(0, 2, 3, 1).reshape(-1, embed_dim)  # [B*H*W, embed_dim]
        image_embeddings_adapted = self.image_adapter(image_embeddings_flat)  # [B*H*W, transformer_dim]
        # Reshape back to [B, transformer_dim, H, W]
        image_embeddings = image_embeddings_adapted.reshape(B, H, W, self.transformer_dim).permute(0, 3, 1, 2)
        
        # Debug: Print intermediate shapes
        if self.training and torch.rand(1).item() < 0.01:  # Only print 1% of the time
            print(f"Adapted embeddings shape: {image_embeddings.shape}")
        
        # Create dummy prompts (no user prompts for automatic segmentation)
        batch_size = x.shape[0]
        sparse_prompt_embeddings = torch.zeros(batch_size, 0, self.transformer_dim, device=x.device)
        dense_prompt_embeddings = torch.zeros(batch_size, self.transformer_dim, H, W, device=x.device)
        
        # Create image positional embeddings
        image_pe = torch.zeros(batch_size, self.transformer_dim, H, W, device=x.device)
        
        # Mask decoder
        masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=False
        )
        
        # Debug: Print intermediate shapes
        if self.training and torch.rand(1).item() < 0.01:  # Only print 1% of the time
            print(f"Decoder output shape: {masks.shape}")
        
        # Final output
        output = self.final_conv(masks)
        
        # Debug: Print intermediate shapes
        if self.training and torch.rand(1).item() < 0.01:  # Only print 1% of the time
            print(f"Final conv output shape: {output.shape}")
            print(f"Before sigmoid range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # Upsample to original image size (256x256)
        if output.shape[-1] != 256 or output.shape[-2] != 256:
            output = F.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)
            if self.training and torch.rand(1).item() < 0.01:
                print(f"After upsampling shape: {output.shape}")
        
        # Apply sigmoid to get proper [0,1] range
        output = torch.sigmoid(output)
        
        if self.training and torch.rand(1).item() < 0.01:
            print(f"After sigmoid range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        return output 