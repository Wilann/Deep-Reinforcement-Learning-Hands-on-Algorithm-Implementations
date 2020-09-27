import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    '''
    Outputs Q-values for every action available 
    Notes:
    - Last layer doesn't have non-linearity applied because Q-values can have any value
    '''
    
    def __init__(self, input_shape, n_actions):
        '''
        input_shape: (4, 84, 84)
        n_actions: 6
        '''
        super(DQN, self).__init__()

        # Convolutional layers 
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), # (4, 84, 84) --> (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # (32, 20, 20) --> (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # (64, 9, 9) --> (64, 7, 7)
            nn.ReLU()
        )

        # Get number of parameters in final convolutional layer 
        conv_out_size = self._get_conv_out(input_shape) # 3136
        
        # Fully-connected layers 
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512), # 3136 --> 512
            nn.ReLU(),
            nn.Linear(512, n_actions) # 512 --> 6
        )

    def _get_conv_out(self, shape):
        '''Applies a convolution to a fake tensor of shape `shape` and returns number of parameters'''
        
        # Apply convolutions to a tensor of zeros 
        o = self.conv(torch.zeros(1, *shape)) # [1, 64, 7, 7]
        
        # Number of parameters in output layer 
        return int(np.prod(o.size())) # 3136 = 1 x 64 x 7 x 7

    def forward(self, x):
        '''
        Feed-forward on a 4D tensor 
        Input Shape: (batch_size, color_channel/stack_of_frames, image_dim_1, image_dim_2)
        '''
        
        # Convolutions
        # Re-shape batch of 3D tensors into batch of 1D vectors 
        conv_out = self.conv(x).view(x.size()[0], -1) # (1, 4, 84, 84) --> (1, 64, 7, 7) --> (1, 3136)
        
        # Linear Part 
        return self.fc(conv_out) # (1, 3136) --> (1, 6)