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




pts = [[[4.4744716, -31.087797, -5.6179533]],    \
         [[4.9958696, -22.94576, -4.198843]],    \
         [[4.9300404, -16.776953, -3.1251352]],    \
         [[  0.010,        0.,            0.       ]],    \
         [[  0. ,         0.       ,   0.       ]],    \
         [[  0.  ,        0.       ,   0.       ]],    \
         [[  0.   ,       0.        ,  0.       ]],
         [[4.9300404, -16.776953, -3.1251352]]]


pts = np.asarray(pts, dtype=np.float32)  
print(pts)
 
print(pts.shape)    # 7 1 3

## not sure if this is correct - take sum and check if sum=0
sum_ = np.sum(np.absolute(pts), axis=2)
print(sum_)   # 8000 1
mask = np.logical_and(sum_ < 50*3, sum_ > 0.0001)
print(mask)
#print(sum_.shape)

old_val = True
#for val in mask:
#    if (old_val == False and val == True):
#        print('!')
#    
#    old_val = val 

test = pts[mask]

print(test.shape)
print(test)

















