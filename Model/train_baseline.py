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
from RGBDataset_10 import RGBDataset
import os

from baseline import Net, loss_function
#conda activate f_frcnn_py36

# loss_function(mu, logvar, mu_next, logvar_next, z_next, z_next_h, action, action_next, action_exp, action_exp_next)

if __name__ == "__main__":
    ## arg parser
    import argparse
    argparser = argparse.ArgumentParser(
            description='training pointFusion using Dagger')
    ## dagger iteration number    
    argparser.add_argument(
        '--iter',
        default='0',
        help='dagger iteration number')

    args = argparser.parse_args()

    #############################################################################
    ## parameters
    batchSize = 200
    testbatch = 200
    
    #npoints = 1900
    nepoch = 50
    load_pretrained = 0
    #weights_path = '/home/zidong/Desktop/nn/pointFusion/6.12_2/cls_model_47.pth'

    outf = 'result'      # output dir for state_dict saving

    raw_data_path = 'dagger_data/ep_'
    test_data_path = '../test_road/ep_'
    
    # this is the number of dagger iteration
    #iteration = 1
    iteration = int(args.iter)
    assert(iteration>=0)
    
    
    
    #############################################################################
    
    if iteration > 0:
        load_pretrained = 1
    ## set up the weight_path for pretrained model from last iteration
    weights_path =  '%s/dagger_%d.pth' % (outf, iteration-1)
    
    ## preproccesing
    print("preproccesing..")

    ## reading image
    print('loading data from', raw_data_path)
    
    ###################### training data ######################################################## 
    print('loading training data.....................')
    data_rgb, data_action_exp = load_data(raw_data_path, 0, iteration+1, 1, 20) # iteration+1

    ## load data and labels
    dataset = RGBDataset(
        data_action_exp=data_action_exp,
        data_rgb = data_rgb)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0)
    
    num_data = data_rgb.shape[0]
    

    #############################################################################
    ## training
    print("training")

    net = Net()
    
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    net.cuda()

    num_batch = num_data / batchSize
    print('num_batch =', num_batch)

    ## load pretrained
    train_loss = []
    if load_pretrained: 
        #'/home/zidong/Desktop/pointnet.pytorch/utils/cls/cls_model_19.pth'
        net.load_state_dict(torch.load(weights_path))
        print('loading pretrained model from ', weights_path)
    else:
        print('no pretrained model, start training..')


    for epoch in range(nepoch):
        scheduler.step()
        total_loss_ephoc = [0.0, 0.0, 0.0, 0.0]
        
        for i, data in enumerate(dataloader, 0):
            
            images, action_exp = data
            images, action_exp = images.cuda(), action_exp.cuda()
            optimizer.zero_grad()
            net = net.train()
            

            ah= net(images)
            ah = torch.squeeze(ah)

            #print(ah.shape, action_exp.shape)
            tloss= loss_function(ah, action_exp)
            tloss.backward()
            optimizer.step()

            total_loss_ephoc[0] += tloss.detach().tolist() #/ target.shape[0]
            
            if i % 10 == 0:
                #print('prediction = ', pred)
                print('[%d: %d/%d] train loss: %f'  % (epoch, i, num_batch, tloss.item()/batchSize)) # average loss
        
        total_loss_ephoc = [x / num_data for x in total_loss_ephoc]
        train_loss.append(total_loss_ephoc)
        print('## ephoc %d train loss:' % epoch, total_loss_ephoc)

    torch.save(net.state_dict(), '%s/dagger_%d.pth' % (outf, iteration))
    print('stored trained model in ', '%s/dagger_%d.pth' % (outf, iteration))


    ## same loss curve


    ## save the total training loss
    train_loss = np.asarray(train_loss)
    loss_log_name2 = '%s/dagger_%d_loss_all.pth' % (outf, iteration)
    np.save(loss_log_name2, train_loss)
    











