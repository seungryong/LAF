import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
from os import listdir
from skimage import io
import models
import datasets
from lossfunc import BCE2d
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='LAFNet (CVPR 2019) stereo confidence training...')
parser.add_argument('--base_lr', type=float, default=3e-3, help="base_lr")
parser.add_argument('--batch_size', type=float, default=4, help="batch_size")
parser.add_argument('--num_epochs', type=int, default=800, help="num_epochs")
parser.add_argument('--step_size_lr', type=int, default=400, help="step_size_lr")
parser.add_argument('--gamma_lr', type=float, default=0.1, help="gamma_lr")
opt = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def train():
    model = models.LAFNet_CVPR2019().to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.base_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size_lr, gamma=opt.gamma_lr)
    
    train_data = datasets.KITTI_train()
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)
    
    for epoch in range(opt.num_epochs):
        count = 0
        train_loss = 0.
        for i, data in enumerate(train_loader):
            cost = data['cost'].to(device)
            disp = data['disp'].to(device)
            imag = data['imag'].to(device)            
            gt_conf = data['gt_conf'].to(device)

            optimizer.zero_grad()
            
            output = model(cost, disp, imag)
            
            loss = BCE2d(output, gt_conf)

            loss.backward()
            optimizer.step()

            train_loss += loss
            count += 1
            
        train_loss = train_loss/count 
        print('Epoch [{}/{}] Train Loss: {:.4f}'.format(epoch+1,opt.num_epochs,train_loss))

        scheduler.step()
        
    torch.save(model.state_dict(),'saved_models/mccnn_networks.pth')

if __name__ == '__main__':
    train()