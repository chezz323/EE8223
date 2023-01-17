import os
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from model.unet import UNet
from model.resunet2 import ResUnet
from dataset import *
from util import *

from loss import DiceLoss, FocalLoss
from metric import DiceCoefficientScore, F1Score


from tqdm import tqdm

class Test(object):
    def __init__(self, model, loss, data_dir, batch_size=1):
        self.model = model
        self.loss = loss
        self.data_dir = data_dir
        self.batch_size = batch_size
        lr = 1e-3
        ckpt_dir = './checkpoint/'+model+'_'+loss
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        net = UNet().to(device) if self.model=='unet' else ResUnet().to(device)
        optim = torch.optim.Adam(net.parameters(), lr=lr)
        self.net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
        
        #data_dir = '../data/test/brain_mri'
        masks_paths = glob(self.data_dir+'/*/*_mask*')
        images_paths = [path.replace("_mask", "") for path in masks_paths]
        self.DataFrame = pd.DataFrame({"images":images_paths,"mask":masks_paths,"diagnosis":[1 if (np.max(cv2.imread(imagepath))>0) else 0 for imagepath in masks_paths]})
        self.DataFrame = self.DataFrame.loc[self.DataFrame['diagnosis']==1]
        
        self.result_dir = './test_result/'+self.model+'_'+self.loss
        if not os.path.exists(self.result_dir):
            os.makedirs(os.path.join(self.result_dir, 'png'))
            os.makedirs(os.path.join(self.result_dir, 'numpy'))
    
    def test(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        net = self.net
        transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
        dataset_test = Dataset(data_dir=self.DataFrame, transform=transform)
        loader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=8)
        num_data_test = len(dataset_test)
        num_batch_test = np.ceil(num_data_test/self.batch_size)
        
        save_df = pd.DataFrame({'id':[],
                                'MODEL':[],
                                'LOSS':[],
                                'DCS':[],
                                'F1-score':[],
                               })
        
        # setrics
        DCS = DiceCoefficientScore()
        metric = F1Score()
        
        # functions
        fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
        fn_denorm = lambda x, mean, std: (x * std) + mean
        fn_class = lambda x: 1.0 * (x > 0.5)
        
        with torch.no_grad():
            net.eval()
            for batch, data in enumerate(loader_test, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)
                output = output[:,1,:,:].unsqueeze(1)
                label = fn_class(label)
                label = label.squeeze()
                if label.ndim==2:
                    label=label.unsqueeze(0)

                # Evaluate by Metric
                dice_coeff = DCS(fn_class(output), label.long())
                f1_score = metric(fn_class(output), label.long())


                print("TEST: BATCH %04d / %04d - DCS %.4f | F1-Score %.4f" % (batch, num_batch_test, dice_coeff, f1_score))

                # Save to Tensorboard 
                label = label.unsqueeze(1)
                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_class(output)#.squeeze().unsqueeze(0)
                #if output.ndim==3:
                #    output = output.unsqueeze(0)
                output = fn_tonumpy(output)

                for j in range(label.shape[0]):
                    id = num_batch_test * (batch - 1) + j

                    plt.imsave(os.path.join(self.result_dir, 'png', '%04d_label.png' % id), label[j].squeeze(), cmap='gray')
                    plt.imsave(os.path.join(self.result_dir, 'png', '%04d_input.png' % id), input[j].squeeze(), cmap='gray')
                    plt.imsave(os.path.join(self.result_dir, 'png', '%04d_output.png' % id), output[j].squeeze(), cmap='gray')

                    temp_df = pd.DataFrame({'id':[id],
                                'MODEL':[self.model],
                                'LOSS':[self.loss],
                                'DCS':[dice_coeff.detach().cpu().numpy()],
                                'F1-score':[f1_score.detach().cpu().numpy()],
                               })
                    save_df = pd.concat([save_df, temp_df], ignore_index=True)
                    

        print("AVERAGE TEST: BATCH %04d / %04d" %(batch, num_batch_test,))
        save_df.to_csv(self.result_dir+'/result_'+self.model+'_'+self.loss+'.csv')