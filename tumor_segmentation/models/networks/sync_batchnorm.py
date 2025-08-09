import torch.nn as nn

# Placeholder implementation for SynchronizedBatchNorm2d.
# For single-GPU/CPU training this behaves the same as nn.BatchNorm2d.
# It can be replaced by a proper SyncBatchNorm implementation if
# multi-GPU synchronisation is required.


class SynchronizedBatchNorm2d(nn.BatchNorm2d):
    """Alias for nn.BatchNorm2d to satisfy SPADE import paths."""

    pass
