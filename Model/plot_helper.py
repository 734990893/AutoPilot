'''
here, we don't specify ep number when loading data
instead we derectly feed in dir name for data
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

import os
##
import math
import numpy.ma as ma

# num_batch per episode
def load_data (path, batch_num_start, batch_num_end):
    ## Training Data
    data_img = []
    data_ctrl = []
    data_pts = []
    data_ctrl_real = []
    
    # images - 100 per batch
    for i in range(batch_num_start, batch_num_end, 1):   # 2 to 43
        ## test if the file exist
        filePath = path + '/CameraRGB/' + str(i) + '.npy'
        if not os.path.exists(filePath):
            print(filePath,'doesnt exist')
            print('its ok, skip')
            continue
        
        imgs = np.load(path  + '/CameraRGB/' + str(i) + '.npy')
        ctrls = np.load(path + '/Control/' + str(i) + '.npy')
        pts = np.load(path  + '/Lidar/' + str(i) + '.npy')
        
        ctrls_real = np.load(path + '/Control_real/' + str(i) + '.npy')
        
        # image
        #imgs_resize = []
        for img in imgs:
            # resize image from 180*320 to 180*180
            img = cv2.resize(img, dsize=(180, 180))   
            data_img.append(img)
        
        #data_img.append(imgs_resize)

        # control
        for c in ctrls:
            data_ctrl.append(c)
        
        for c2 in ctrls_real:
            data_ctrl_real.append(c2)
            
            

    ## control: [throttle, steer, brake, speed]
    ctrl = np.asarray(data_ctrl, dtype=np.float32)
    ctrl = ctrl.reshape(ctrl.shape[0],  -1)
    print('ctrl shape: ', ctrl.shape)                               # (N, 4)
    labels = ctrl[:, 1]  # steering                              # (N, 1)
    
    ctrl_real = np.asarray(data_ctrl_real, dtype=np.float32)
    ctrl_real = ctrl_real.reshape(ctrl.shape[0],  -1)
    print('ctrl_real shape: ', ctrl_real.shape)                               # (N, 4)
    ctrl_real = ctrl_real[:, 1]
    
    
    # TODO
    #for label in labels:
    #    if math.isnan(label):
    
    
    ## image
    image = np.asarray(data_img, dtype=np.float32)              # (N, 180, 180, 3)                          
    # convert image to [0, 1]
    image /= 255
    # convert image to [-1, 1]
    inputs = image *2 -1
    inputs = np.transpose(inputs, (0, 3, 1, 2))                  # (N, 3, 180, 180)
    print('image shape: ', inputs.shape) 
    
    ## 3d point cloud
    #points = np.asarray(data_pts, dtype=np.float32)                 # (N, 1900, 3)
    #print('points shape: ', points.shape) 
    
    
    ## check if labels are legit
    # solve the problem of having nan stored in steering right after respawn to new position
    mask2 = np.logical_not(np.isnan(labels))
    inputs = inputs[mask2]
    labels = labels[mask2]
    #points = points[mask2]
    ctrl_real = ctrl_real[mask2]
    
    print('ctrl shape after mask nan: ', labels.shape) 
    print('image shape after mask nan: ', inputs.shape) 
    #print('points shape after mask nan: ', points.shape) 
    print('ctrl_real shape after mask nan: ', ctrl_real.shape) 
    
    data_rgb = torch.from_numpy(inputs)     # image
    data_labels = torch.from_numpy(labels)  # steering
    #data_points = torch.from_numpy(points)  # Lidar detection 3d points
    data_ctrl_real = torch.from_numpy(ctrl_real) 

    ##
    data_points = []
    return data_rgb, data_points, data_labels, data_ctrl_real


# usage
# path = '/home/zidong/Desktop/nn/epn'

# data_rgb, data_points, data_labels, data_ctrl_real = load_data(path, 0, 4, 0, 44)





















