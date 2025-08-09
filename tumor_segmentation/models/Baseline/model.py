import torch
import torch.nn as nn
from models.base_model import BaseModel


class BaselineModel(BaseModel):
    """
    Baseline model that implements thresholding like in example.py.
    Uses a learnable threshold parameter to keep backpropagation working,
    but essentially does simple thresholding on input images.
    """

    def __init__(self, threshold=50.0, lr=1e-3, weight_decay=1e-5):
        super().__init__(lr=lr, weight_decay=weight_decay)

        # Learnable threshold parameter (won't really learn, but keeps backprop working)
        # Note: threshold is in [0,255] range to match API format
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))

    def forward(self, x):
        """
        Apply thresholding like in example.py:
        segmentation = (img < threshold).astype(np.uint8) * 255

        But adapted for torch tensors with [0-1] range (normalized).
        Uses differentiable approximation for gradients.
        """
        # Convert RGB to grayscale (simple average)
        # x is [B, 3, H, W] with values in [0-1] range (normalized), convert to [B, 1, H, W]
        grayscale = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]

        # Convert threshold from [0-255] to [0-1] range to match input
        threshold_normalized = self.threshold / 255.0

        # Use differentiable approximation of thresholding
        # This creates a smooth approximation of (img < threshold)
        # Working with [0-1] values (normalized)
        steepness = (
            25.0  # Controls how sharp the transition is (larger for [0-1] range)
        )
        threshold_diff = (
            threshold_normalized - grayscale
        )  # Positive when pixel < threshold

        # Apply sigmoid: outputs ~1 when pixel < threshold, ~0 when pixel > threshold
        output = torch.sigmoid(threshold_diff * steepness)

        return output
