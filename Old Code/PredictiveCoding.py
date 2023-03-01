'''
This is a simple implementation of the predictive coding model of perception.
'''

# Import the necessary modules, I'm going to use torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the model
class PredictiveCoding(nn.Module):
    '''
    This is a simple implementation of the predictive coding model of perception.
    It is going to use conv2d layers to implement the model.
    '''

    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        '''
        This is the initialization function for the model.
        '''
        super(PredictiveCoding, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.unfold = nn.Unfold(kernel_size, stride, padding)

    def forward(self, x):
        '''
        This is the forward function for the model.
        x is an image of size (batch_size, channels, height, width)
        '''

        patches = self.unfold(x) # (batch_size, channels*kernel_size*kernel_size, num_patches)

        return patches


