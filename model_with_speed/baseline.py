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

        self.fc1 = nn.Linear(180*180*3, 576)
        self.fc2 = nn.Linear(576, 240)
        self.fc3 = nn.Linear(240, 128)
        self.fc4 = nn.Linear(128, 1)
           

    def forward(self, x, mode='default'):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        ah = self.fc4(x)
        
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







