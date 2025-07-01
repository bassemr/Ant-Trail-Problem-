"""
This script defines a simple neural network model using PyTorch. 
The model consists of two linear layers and applies a ReLU activation function in between. 
It is designed for tasks such as classification or regression, where input data is transformed into output predictions.
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F



class Model(nn.Module):
    """
    A neural network model with one hidden layer.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the model with two linear layers.
        
        Parameters:
        - input_size: Number of input features.
        - hidden_size: Number of neurons in the hidden layer.
        - output_size: Number of output classes or values.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Parameters:
        - x: Input tensor.
        
        Returns:
        - Output tensor after passing through the network.
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
