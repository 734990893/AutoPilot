"""
plot 2: image reconstruction

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
    default='5',
    help='dagger iteration number')
    
argparser.add_argument(
    '--step',
    default='5',
    help='dagger iteration number')

args = argparser.parse_args()  
    




#############################################################################
## parameters
num_steps=int(args.step)

batchSize = 40#100
npoints = 1900
nepoch = 50
load_pretrained = 0

outf = 't_result'      # output dir for state_dict saving

test_path = '../test_road/ep_'
train_path = './dagger_data/ep_'
#train_path = './dagger_data/t_ep_'

data_path = './dagger_data/ep_7'

# this is the number of dagger iteration
iteration = args.iter


#weights_path =  './result/dagger_%d.pth' % (int(iteration) - 1) # ep_5 data load dagger_4 weight
#'%s/%s.pth' % (outf, iteration)
weights_path =  './result/epn5_i7.pth'

#xlim = [0,2000]
#xlim = [850,960]

#############################################################################

## preproccesing
print("preproccesing..")

##################### training data ############################################
print('loading training data..................... iteration:', iteration)
data_rgb, data_points, data_action_exp, data_action_real = load_data(data_path, 0, 3)

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
## total training loss

running_loss = [0.0, 0.0, 0.0, 0.0, 0.0]

## place holder to store values
empty = []
for i in range(num_steps):
    empty.append([])
empty = [[],[],[],[],[]]
#fig = plt.figure()
sample = [[],[],[],[],[]]
#z_sample = empty

num_sample = 4

z_sample = torch.zeros(num_sample, num_steps+1, num_data-num_steps, 240)   # 10 6 295 240
for j in range(num_sample):     

    pred = empty.copy()
    #z = empty
    
    for i, data in enumerate(dataloader, 0):
        net = net.eval()
        images, action_exp, action_real = data
        images, action_exp, action_real = images.cuda(), action_exp.cuda(), action_real.cuda()
        images = torch.transpose(images, 0, 1)
        #print(images.shape) #torch.Size([11, 20, 3, 180, 180])

        with torch.no_grad():
            #mu, logvar, a_list, zh_list = net(images[0], action_real)
            a_list, zh_list = net(images[0], [], mode ='recon')
            a_list = torch.squeeze(a_list)
            
            #print(a_list.shape, zh_list.shape)  # s, b  /  s, b, 240
            
            if i==0:
                z = zh_list
            else:
                z = torch.cat((z, zh_list), dim=1)
            
        for i in range(num_steps):
            pred[i].extend(a_list[i+1,:].detach().tolist())
        
    print('z', z.shape) #[6, 295, 240]
    z_sample[j] = z
    #print('pred', np.shape(pred))   # 5, 295
    for i in range(num_steps):
        sample[i].append(pred[i])
        
print('done')
print('z_sample',z_sample.shape)

sample = np.asarray(sample)
#print(sample.shape) # 5, 10, 295


#z_sample = z_sample.numpy()
#print('z_sample',z_sample.shape)    # 10 6 295 240
#z_sample = np.mean(z_sample, axis=0)#6 295 240
#print('z_sample',z_sample.shape)

v = empty.copy()
m = empty.copy()
for i in range(num_steps):
    v[i] = np.var(sample[i], axis=0)
    m[i] = np.mean(sample[i], axis=0)
    '''
    for j in range(i+1):
        m[i] = np.concatenate([[0], m[i]])
        v[i] = np.concatenate([[0], v[i]])
    '''

action_real = data_action_real.numpy()
action_exp = data_action_exp.numpy()
action_real = np.reshape(action_real, (-1,))
action_exp = np.reshape(action_exp, (-1,))


tr = np.arange(0, action_real.shape[0], 1)
te = np.arange(0, action_exp.shape[0], 1)
t = empty.copy()
for i in range(num_steps):
    t[i] = np.arange(0, m[i].shape[0], 1)


####################
## replay - current image and decoded image


rows = 1 + num_sample # 5
cols = num_steps+2 # img img_r img_1 .. img_5


for i, img in enumerate(data_rgb, 0):
    if(i<240) and 1:
        continue

    fig = plt.figure()
    plt.gca().set_title('data: frame '+str(i))
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # X AXIS -BORDER
    plt.gca().spines['bottom'].set_visible(False)
    # Y AXIS -BORDER
    plt.gca().spines['left'].set_visible(False)
    
    ## plot real images seen
    for s in range(num_steps+1):
        fig.add_subplot(rows, cols, s+1)
        plt.gca().set_title('input'+str(s))
        img = data_rgb[i+s]
        img = (img +1) / 2
        plt.imshow(img.permute(1, 2, 0))
    
    
    ## plot reconstructed image
    '''
    # z_sample: 6 295 240
    for s in range(num_steps+1):
        fig.add_subplot(rows, cols, cols+s+1)
        plt.gca().set_title('recon'+str(s))
        z = torch.from_numpy(z_sample[s, i, :]).cuda()
        img_recon = net.decode(z)   # 1 3 180 180
        img_recon = (img_recon +1) / 2
        img_recon = torch.squeeze(img_recon)    # 3 180 180
        #print(img_recon.shape)
        plt.imshow(img_recon.detach().cpu().permute(1, 2, 0))
    '''
    # z_sample: pytorch 4 6 295 240
    for j in range(num_sample):
        for s in range(num_steps+1):
            fig.add_subplot(rows, cols, (j+1)*cols+s+1)
            #plt.gca().set_title('recon'+str(s))
            z = z_sample[j, s, i, :].cuda()
            img_recon = net.decode(z)   # 1 3 180 180
            img_recon = (img_recon +1) / 2
            img_recon = torch.squeeze(img_recon)    # 3 180 180
            #print(img_recon.shape)
            plt.imshow(img_recon.detach().cpu().permute(1, 2, 0))
        
    
    plt.show()


    
    
