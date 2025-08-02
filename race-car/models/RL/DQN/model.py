import torch
from torch import nn
from torch.nn import functional as F

class DQN(nn.Module):
    def __init__(self, input_dim: int = 21, output_dim: int = 1):
        super(DQN, self).__init__()
        self.inputlayer = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = F.relu(self.inputlayer(x))
        x = F.relu(self.layer2(x))
        return self.output(x)
        