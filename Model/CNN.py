'''
Embeded Predictive Network
simplified version of E2C paper


10 time step prediction during training
- using linear instead of baysian linear
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import cv2
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

import random
from load_data import load_data
import os
import math
#conda activate f_frcnn_py36

        
#####################################
class Net(nn.Module):

    def __init__(self, num_steps=10):
        super(Net, self).__init__()
        
        self.num_steps = num_steps
        
        # 3 input image channel, 64 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 64, 3)    # 180 * 180 * 64
        self.conv2 = nn.Conv2d(64, 64, 3)   # 90
        self.conv3 = nn.Conv2d(64, 64, 3)   # 45
        self.conv4 = nn.Conv2d(64, 64, 3)   # 9
        self.conv5 = nn.Conv2d(64, 64, 3)   # 3
        
        # action decoder
        self.fc1 = nn.Linear(576, 240)
        self.fc2 = nn.Linear(240, 128)
        self.fc3 = nn.Linear(128, 1)
        
   
    
    def encode(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv1(x)  
        x = F.max_pool2d(F.leaky_relu(x), (2, 2)) # 90 * 90 * 64
        x = F.max_pool2d(F.leaky_relu(self.conv2(x)), 2)  # 45 * 45 * 64
        x = F.max_pool2d(F.leaky_relu(self.conv3(x)), 2)  # 9 * 9 * 64
        x = F.max_pool2d(F.leaky_relu(self.conv4(x)), 2)  # 3 * 3 * 64
        x = F.max_pool2d(F.leaky_relu(self.conv5(x)), 2)  # 3 * 3 * 64        
        #print(x.shape)  # torch.Size([100, 64, 3, 3])
        
        x = x.view(-1, self.num_flat_features(x))
        
        #x = F.leaky_relu(self.fc01(x))
        #x = F.leaky_relu(self.fc02(x))

        return x
   

    # TODO action decoder too small?
    def action_decode(self, z):
        #print(z.shape)
        z = F.leaky_relu(self.fc1(z))
        z = F.leaky_relu(self.fc2(z))
        
        z = self.fc3(z)
        #z = self.sigmoid(self.fc3(z)) # [0, 1]
        return z  # map to [1, -1]
    
   
    # input: x_t, x_t+1
    # x0: 50, 3, 180, 180
    # u 50, 1
    
    # q_z -> z from t+1 to t+T decoded from corresponding img <- NA
    
    # pass in x => s b 3 180 180
    def forward(self, x, mode='default'):
        #if self.training:
        z = self.encode(x)
        ah = self.action_decode(z)
        
        return ah


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def loss_function(ah, action_exp):
    MSEa = F.mse_loss(ah, action_exp, reduction='sum') 
    return MSEa







