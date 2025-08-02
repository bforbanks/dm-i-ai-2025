import torch
from torch import nn

class DQN(nn.Module):
    '''
        Deep Q-Network (DQN) model.
        
        The model consists of two hidden layers with ReLU activation functions at the moment.
        
        Args:
            input_dim (int): Dimension of the input features, default is 21.
            output_dim (int): Dimension of the output actions, default is 5.
    '''
    def __init__(self, input_dim: int = 21, output_dim: int = 5):
        super(DQN, self).__init__()
        self.inputlayer = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, output_dim)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.inputlayer(x))
        x = self.relu(self.layer2(x))
        return self.output(x)
        