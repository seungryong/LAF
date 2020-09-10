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

class KITTI_train(Dataset):
    def __init__(self):
        self.image_path = 'KITTI-data/2012/left'
        image_names = sorted(listdir(self.image_path))
        image_names = [x for x in image_names if x.find("png") != -1 if not x.startswith('.')] 
        
        self.dataset_path = 'KITTI-data/2012/mccnn'
        dataset_names = sorted(listdir(self.dataset_path))
        disp_names = [x for x in dataset_names if x.find("png") != -1 if not x.startswith('.')] 
        cost_names = [x for x in dataset_names if x.find("bin") != -1 if not x.startswith('.')]
        
        self.gt_disp_path = 'KITTI-data/2012/training/disp_occ'
        gt_disp_names = sorted(listdir(self.gt_disp_path))
        gt_disp_names = [x for x in gt_disp_names if x.find("png") != -1 if not x.startswith('.')] 
               
        self.image_names = image_names[:20]
        self.disp_names = disp_names[:20]
        self.cost_names = cost_names[:20]
        self.gt_disp_names = gt_disp_names[:20]
        
    def __len__(self):
        return 20

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        imag_name = os.path.join(self.image_path,self.image_names[idx])
        imag = io.imread(imag_name)
        imag = torch.Tensor(imag.astype(np.float32)/256.)
        imag = imag.transpose(0,1).transpose(0,2)
        
        ch, hei, wei = imag.size() 
        
        cost_name = os.path.join(self.dataset_path,self.cost_names[idx])
        cost = np.memmap(cost_name, dtype=np.float32, shape=(228, hei, wei))
        cost = torch.Tensor(cost)
        cost[cost==0] = cost.max()
        
        disp_name = os.path.join(self.dataset_path,self.disp_names[idx])
        disp = io.imread(disp_name)
        disp = torch.Tensor(disp.astype(np.float32)/256.).unsqueeze(0)
        
        gt_disp_name = os.path.join(self.gt_disp_path,self.gt_disp_names[idx])
        gt_disp = io.imread(gt_disp_name)
        gt_disp = torch.Tensor(gt_disp.astype(np.float32)/256.).unsqueeze(0)
        
        gt_conf = (torch.abs(disp-gt_disp) <= 3).type(dtype=torch.float)
        gt_conf[gt_disp==0] = -1
                        
        new_hei = 128
        new_wei = 256
        
        top = np.random.randint(0,hei-new_hei)
        left = np.random.randint(0,wei-new_wei)

        cost = cost[:,top:top+new_hei,left:left+new_wei]
        disp = disp[:,top:top+new_hei,left:left+new_wei]/228.
        imag = imag[:,top:top+new_hei,left:left+new_wei]
        gt_conf = gt_conf[:,top:top+new_hei,left:left+new_wei]
        
        return {'cost': cost, 'disp': disp, 'imag': imag, 'gt_conf': gt_conf}