#!/usr/bin/env python

'''
verify and visualize the data collected from CARLA simulator

'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

import random


folder = 'data6.5'  # '_out_data'
episode = 1
batch_num = 0

RGB_NAME = '/home/zidong/Desktop/CARLA_0.9.5/data_collector/%s/ep_%d/CameraRGB/%d.npy' % (folder, episode, batch_num)
LIDAR_NAME = '/home/zidong/Desktop/CARLA_0.9.5/data_collector/%s/ep_%d/Lidar/%d.npy' % (folder, episode, batch_num)
CTRL_NAME = '/home/zidong/Desktop/CARLA_0.9.5/data_collector/%s/ep_%d/Control/%d.npy' % (folder, episode, batch_num)

print('loading.. '+ RGB_NAME)

# load the numpy arrays
rgbs = np.load(RGB_NAME)
#sems = np.load(SEMSEG_NAME)
lidar = np.load(LIDAR_NAME)
control = np.load(CTRL_NAME)

print('Shapes: RGB: ', rgbs.shape)      # 180 * 320
print('Data types: RGB: ', rgbs.dtype)  # uint8

print('Shapes: Lidar: ', lidar.shape)   # (1000, 5000, 1, 3)
print('Data types: Lidar: ', lidar.dtype)

lidar = lidar.reshape((lidar.shape[0], lidar.shape[1], 3))


#assert rgbs.shape[1:] == (240, 320, 3)  # the RGB images should be HEIGHT x WIDTH x 3 in shape
#assert sems.shape[1:] == (240, 320, 1)  # the sematic segmentation ground truth should be the same shape as the depth maps
                                        # note that HEIGHT and WIDTH are defined in `mc_buffered_save.py` as
                                        # WINDOW_WIDTH and WINDOW_HEIGHT respectively

assert rgbs.dtype == np.uint8  # the RGB images have 8-bit pixel values
#assert sems.dtype == np.uint8  # the semantic segmentation ground truth only has 14 distinct values, so 8-bit ints are enough

# choose 16 numbers between 0 and 999 to choose 16 images to display (note that the last arrays in any episode
# will have less than 1000 images, so modify the call to `random.randint` accordingly)
indices = random.sample(range(0, 898), 16)


## for image data
def plot_16(arr, indices, sensor_name, fig_title=None):
    """Plot 16 images in `arr` with the indices in `indices`.
    `sensor_name` is one of 'rgb' and 'semseg', describing the type of sensor data.
    """
    
    ## for image data
    h, w, num_channels = arr.shape[1:]
    display_array = np.empty(shape=(4 * h, 4 * w, num_channels),
                             dtype=arr.dtype)

    #'''
    for r in range(4):
        for c in range(4):
            display_array[r * h: (r + 1) * h,
                          c * w: (c + 1) * w] = arr[indices[c + 4*r]]
            #print(arr[indices[c + 4*r]].shape)
            #plt.imshow(arr[indices[c + 4*r]], aspect='auto')
    '''
    r = 3
    c = 3
    display_array[r * h: (r + 1) * h,
                  c * w: (c + 1) * w] = arr[indices[c + 4*r]]
    '''
    
    plt.figure(figsize=(16, 12))
    if fig_title is not None:
        plt.title(fig_title)
    
    if sensor_name == 'rgb':
        display_array = cv2.cvtColor(display_array, cv2.COLOR_BGR2RGB)  # convert images from BGR to RGB using opencv
        plt.imshow(display_array, aspect='auto')
    elif sensor_name == 'semseg':
        plt.imshow(display_array[:, :, 0], cmap='tab20', aspect='auto')
        plt.colorbar()
        
    else:
        raise ValueError('Unsupported sensor type ', sensor_name)
    
    #plt.show()
    
    
## for Lidar data
def plot_3dpts(arr, indices, sensor_name, fig_title=None):
    
    '''
    ## for image data
    h, w, num_channels = arr.shape[1:]
    display_array = np.empty(shape=(4 * h, 4 * w, num_channels),
                             dtype=arr.dtype)
    #plt.figure(figsize=(16, 12))
    for r in range(4):
        for c in range(4):
            display_array[r * h: (r + 1) * h,
                          c * w: (c + 1) * w] = arr[indices[c + 4*r]]
            #print(arr[indices[c + 4*r]].shape)
            #plt.imshow(arr[indices[c + 4*r]], aspect='auto')
    '''
    r = 3
    c = 3
    ## for Lidar data
    points = arr[indices[c + 4*r]]
    
    print(points.shape) # (5000, 3)
    
    
    fig = plt.figure(figsize=(16, 12))
    if fig_title is not None:
        plt.title(fig_title)
    
    if sensor_name == 'lidar':
        ## plot 3d point cloud
        ax = fig.add_subplot(111, projection='3d')
        
        print(points)
        x = points[:,0]
        print(x.shape)
        y = points[:,1]
        z = points[:,2] 
        
        ax.scatter(x, y, z, c='r', marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        #print(arr[indices[c + 4*r]])
        
    else:
        raise ValueError('Unsupported sensor type ', sensor_name)
    
    #plt.show()

## for 1 Lidar data
def plot_3dpts_1(arr, index, sensor_name, fig_title=None):
    
    '''
    ## for image data
    h, w, num_channels = arr.shape[1:]
    display_array = np.empty(shape=(4 * h, 4 * w, num_channels),
                             dtype=arr.dtype)
    #plt.figure(figsize=(16, 12))
    for r in range(4):
        for c in range(4):
            display_array[r * h: (r + 1) * h,
                          c * w: (c + 1) * w] = arr[indices[c + 4*r]]
            #print(arr[indices[c + 4*r]].shape)
            #plt.imshow(arr[indices[c + 4*r]], aspect='auto')
    '''
    r = 3
    c = 3
    ## for Lidar data
    points = arr[index]
    
    print(points.shape) # (5000, 3)
    
    
    #fig = plt.figure(figsize=(16, 12))
    fig = plt.figure(figsize=plt.figaspect(0.5))
    if fig_title is not None:
        plt.title(fig_title)
    
    if sensor_name == 'lidar':
        ## plot 3d point cloud
        ax = fig.add_subplot(111, projection='3d')
        
        print(points)
        x = points[:,0]
        print(x.shape)
        y = points[:,1]
        z = points[:,2] 
        
        ax.scatter(x, y, z, c='r', marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        #print(arr[indices[c + 4*r]])
        
    else:
        raise ValueError('Unsupported sensor type ', sensor_name)
        
#
## 6.6 plt lidar pts without [0,0,0]
def plot_3dpts_2(arr, index, sensor_name, fig_title=None):

    r = 3
    c = 3
    ## for Lidar data
    points = arr[index]
    
    print(points.shape) # (5000, 3)
    
    
    #fig = plt.figure(figsize=(16, 12))
    fig = plt.figure(figsize=plt.figaspect(0.5))
    if fig_title is not None:
        plt.title(fig_title)
    
    if sensor_name == 'lidar':
        ## plot 3d point cloud
        ax = fig.add_subplot(111, projection='3d')
        
        #print(points)   # (8000, 3)
        
        ## 1
        #num = np.count_nonzero(points)
        #print(num/3)
        #points = points[0:int(num/3), :]
        
        ## 2
        points = points[0:1900,:]
        points = np.unique(points, axis=0) 
        
        #for pt in points:
            #if(pt)
        
        print('after filter 0: ', points.shape)
        
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]
        
        ax.scatter(x, y, z, c='r', marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        #print(arr[indices[c + 4*r]])
        
    else:
        raise ValueError('Unsupported sensor type ', sensor_name)
        
        
##plotting random pictures
#plot_16(rgbs, indices, 'rgb', 'RGB images')
#plot_3dpts(lidar, indices, 'lidar', 'lidar 3d point cloud')


indices = np.arange(0,16,1)

plot_16(rgbs, indices, 'rgb', 'RGB images')


plot_3dpts_1(lidar, indices[15], 'lidar', 'lidar 3d point cloud')
plot_3dpts_2(lidar, indices[15], 'lidar', 'lidar 3d point cloud_2')

#plot_3dpts_1(lidar, 14, 'lidar', 'lidar 3d point cloud')
#plot_3dpts_2(lidar, 14, 'lidar', 'lidar 3d point cloud_2')


print("control: ", control[indices[15]])


plt.show()

