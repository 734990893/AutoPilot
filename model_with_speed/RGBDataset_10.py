from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement


class RGBDataset(data.Dataset):
    def __init__(self, data_action_exp, data_rgb):
        #self.npoints = npoints
        #self.data_points = data_points
        self.data_action_exp = data_action_exp
        self.data_rgb = data_rgb                # 2000 3 180 180

        
    def __getitem__(self, index):
        
        # current
        action_exp = self.data_action_exp[index]    # 50, 11, 1
        image = self.data_rgb[index]                # 50, 11, 3, 180, 180
        
        return image, action_exp

    def __len__(self):
        return len(self.data_action_exp)
    
  
