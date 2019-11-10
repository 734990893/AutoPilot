'''
created 6.7
Used to preprocess the point cloud data collected from CARLA
Issues to solve:
    - pts collected have diff size for each image
    - pts with large value (e35) and small values (e-35) should be eliminated

    - sort them? not sure
'''


#import torch
#import torch.nn as nn
#import torch.nn.functional as F

#import torch.optim as optim

import cv2
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

import random



# num_batch per episode
def load_pts (path, episode_start, episode_end, batch_num_start, batch_num_end):
    ## Training Data
    data_pts = []
    
    # episodes
    for ep in range(episode_start, episode_end, 1):
        # images - 100 per batch
        for i in range(batch_num_start, batch_num_end, 1):   # 2 to 43

            pts_batch = np.load(path + str(ep) + '/Lidar/' + str(i) + '.npy')
            
            for pts in pts_batch:
                # filter out points with [0, 0, 0]
                #pts_perBatch_filter = []
                
                #print(pts)
                #print(pts.shape)    # 8000 1 3
                
                ## not sure if this is correct - take sum and check if sum=0
                sum_ = np.sum(np.absolute(pts), axis=2)
                #print(sum_.shape)   # 8000 1
                mask = np.logical_and(sum_ < 50*3, sum_ > 0.0001)
                #print(mask)
                #print(sum_.shape)
                
                old_val = True
                #for val in mask:
                #    if (old_val == False and val == True):
                #        print('!')
                #    
                #    old_val = val 

                pts_filter = pts[mask]
                
                #print(test.shape)

                
                
                ## 2
                #pts_perBatch = np.unique(pts_perBatch, axis=0) 
                ## note: can't convert it to numpy array if each batch have different number of points
                
                data_pts.append(pts_filter)
                #data_pts = np.append(data_pts, pts_perBatch)


    ## 3d point cloud
    #print(data_pts) 
    points = np.asarray(data_pts, dtype=np.float32)                 # (N, 8000, 1, 3)
    points = points.reshape(points.shape[0], points.shape[1], -1)   # (N, 8000, 3)
    print(points.shape) 

    data_points = points
    #data_points = torch.from_numpy(points)  # Lidar detection 3d points
    
    return  data_points

# usage
path = '/home/zidong/Desktop/nn/data/ep_'

data_points = load_pts(path, 0, 1, 0, 44)

print(data_points.shape)




















