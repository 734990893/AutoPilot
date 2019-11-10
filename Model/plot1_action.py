"""
plot1: action (predicted) and confidence interval plot
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

import random
import argparse


from plot_helper import load_data
from RGBDataset_10 import RGBDataset
from EPN_10 import Net

argparser = argparse.ArgumentParser(
    description='C')
        
argparser.add_argument(
    '--iter',
    default='7',
    help='dagger iteration number')
    
argparser.add_argument(
    '--step',
    default='50',
    help='dagger iteration number')

args = argparser.parse_args()  
    

#############################################################################
## parameters
num_steps=int(args.step)

batchSize = 20#100

num_samples = 20

load_pretrained = 0
nepoch = 50
outf = 't_result'      # output dir for state_dict saving

test_path = '../test_road/ep_'
train_path = './dagger_data/ep_'
#train_path = './dagger_data/t_ep_'

data_path = './dagger_data/ep_7'

# this is the number of dagger iteration
iteration = args.iter


#weights_path =  './result/dagger_%d.pth' % (int(iteration) - 1) # ep_5 data load dagger_4 weight
#'%s/%s.pth' % (outf, iteration)
weights_path =  './result/dagger_6.pth'

#xlim = [0,2000]
#xlim = [850,960]

#############################################################################

## preproccesing
print("preproccesing..")

##################### training data ############################################
print('loading training data..................... iteration:', iteration)
data_rgb, data_points, data_action_exp, data_action_real = load_data(data_path, 0, 20)

## load data and labels
dataset = RGBDataset(
    data_action_exp=data_action_exp,
    data_rgb = data_rgb,#rgb_10,
    data_action_real = data_action_real,
    num_steps = num_steps)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchSize,
    shuffle=False,
    num_workers=0)

#data_rgb.shape # 300, 3, 180, 180

num_data = data_rgb.shape[0]

print('finished loading data, length =', len(data_action_exp))


#############################################################################
## training
print("training data")

net = Net(num_steps=num_steps)

#optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
net.cuda()

num_batch = num_data  / batchSize
print('num_batch =', num_batch)

## load pretrained
net.load_state_dict(torch.load(weights_path))
print('loading pretrained model from ', weights_path)

########################################################################################### training data ep_5



sample = torch.zeros(num_samples, num_steps+1, num_data-num_steps)

for j in range(num_samples):
    
    pred = torch.zeros(num_steps+1, num_data-num_steps)

    for i, data in enumerate(dataloader, 0):
        net = net.eval()
        images, action_exp, action_real = data
        images, action_exp, action_real = images.cuda(), action_exp.cuda(), action_real.cuda()
        images = torch.transpose(images, 0, 1)
        with torch.no_grad():
            #mu, logvar, a_list, zh_list = net(images[0], action_real)
            a_list = net(images[0], [])
            a_list = torch.squeeze(a_list)  # s+1, b

            pred[:, i*batchSize: (i+1)*batchSize] = a_list ##?
    sample[j] = pred
    
#sample = np.asarray(sample)
#sample = sample[:, 1:].numpy()
sample = sample.numpy()
print(sample.shape) # 50 51 150

v = [[],[],[],[],[]]
m = [[],[],[],[],[]]

#v = torch.zeros(num_steps, num_data-num_steps)
#m = torch.zeros(num_steps, num_data-num_steps)
for i, s in enumerate(np.asarray([1, 5, 10, 20, 50])): #range(num_steps):
    print(i)
    v[i] = np.var(sample[:, s], axis=0)
    m[i] = np.mean(sample[:, s], axis=0)
    #'''
    for j in range(s):
        m[i] = np.concatenate([[0], m[i]])
        v[i] = np.concatenate([[0], v[i]])
    #'''
    #print(m[i])
    #print(m[i].shape[0])

action_real = data_action_real.numpy()
action_exp = data_action_exp.numpy()
action_real = np.reshape(action_real, (-1,))
action_exp = np.reshape(action_exp, (-1,))


tr = np.arange(0, action_real.shape[0], 1)
te = np.arange(0, action_exp.shape[0], 1)

#t = torch.zeros(num_steps, num_data-num_steps)
t = [[],[],[],[],[]]
for i, s in enumerate(np.asarray([1, 5, 10, 20, 50])):#range(num_steps):
    t[i] = np.arange(0, m[i].shape[0], 1)

fig = plt.figure()
############################## plot 1 ###############################################
#fig.add_subplot(3, 1, 1)

#print('action_real', action_real.shape, t.shape)

color = ['y', 'c', 'r', 'm', 'g']
for i, s in enumerate(np.asarray([1, 5, 10, 20, 50])):#[1, 5, 10, 20, 50]:

    plt.plot(t[i], m[i], alpha=1, color = color[i], label = 'pred_'+str(s))
    low_CI = m[i] - 2*v[i] 
    upper_CI = m[i] + 2*v[i] 
    plt.fill_between(t[i], low_CI, upper_CI, color = color[i], alpha = 0.4, label = 'pred_'+str(s)+': 95% confidence')


plt.plot(tr, action_real, alpha=1, color = 'b', label = 'act_real')
plt.plot(te, action_exp, alpha=1, color = 'k', label = 'expert')


#plt.gca().set_ylim([-1,1])
plt.gca().set_title(weights_path + ', data - %s' % data_path)
plt.legend(loc = 'best')
plt.xlabel('frame number')
plt.ylabel('steering angle')
#plt.gca().set_xlim(xlim)

plt.gca().set_ylim([-1, 1])

plt.show()
#'''

    
    
