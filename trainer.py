import os
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import PIL

from model.unet import UNet
from model.resunet2 import ResUnet
from dataset import *
from util import *

from loss import DiceLoss, FocalLoss


from tqdm.notebook import tqdm


class Trainer(object):
    def __init__(self, model, loss, data_dir, lr, batch_size, num_epoch, ckpt_dir, log_dir, result_dir, train_continue):
        self.model = model
        self.loss = loss
        self.data_dir = data_dir
        self.lr = lr
        self.batch_size = batch_size
        self.num_epoch = int(num_epoch)
        self.ckpt_dir = ckpt_dir+'/'+model+'_'+loss
        self.log_dir = log_dir+'/'+model+'_'+loss
        self.result_dir = result_dir+'/'+model+'_'+loss
        self.train_continue = train_continue
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("learning rate: %.4e" % lr)
        print("batch size: %d" % batch_size)
        print("number of epoch: %d" % num_epoch)
        print("data dir: %s" % data_dir)
        print("ckpt dir: %s" % ckpt_dir)
        print("log dir: %s" % log_dir)
        print("result dir: %s" % result_dir)
        
        if not os.path.exists(self.result_dir):
            os.makedirs(os.path.join(self.result_dir, 'png'))
            os.makedirs(os.path.join(self.result_dir, 'numpy'))

        #data_dir = '../data/brain_mri'
        masks_paths = glob(self.data_dir+'/*/*_mask*')
        images_paths = [path.replace("_mask", "") for path in masks_paths]
        DataFrame = pd.DataFrame({"images":images_paths,"mask":masks_paths,"diagnosis":[1 if (np.max(cv2.imread(imagepath))>0) else 0 for imagepath in masks_paths]})
        DataFrame = DataFrame.loc[DataFrame['diagnosis']==1]
        
        self.train_data, self.val_data = train_test_split(DataFrame,test_size=0.2,random_state=42)       

    def train(self):
        transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

        dataset_train = Dataset(data_dir=self.train_data, transform=transform)
        loader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=8)
    
        dataset_val = Dataset(data_dir=self.val_data, transform=transform)
        loader_val = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=8)
    
        num_data_train = len(dataset_train)
        num_data_val = len(dataset_val)
    
        num_batch_train = np.ceil(num_data_train / self.batch_size)
        num_batch_val = np.ceil(num_data_val / self.batch_size)
        
        tf = transforms.ToPILImage()
        
        # Generate Network
        if self.model == 'unet':
            net = UNet().to(self.device)
        elif self.model == 'resunet':
            net = ResUnet().to(self.device)
        
        # Loss function
        if self.loss == 'DL':
            fn_loss = DiceLoss().to(self.device)
            sig = nn.Sigmoid()
        elif self.loss == 'FL':
            fn_loss = FocalLoss().to(self.device)
            sig = nn.Sigmoid()
        elif self.loss == 'BCE':
            fn_loss = nn.BCEWithLogitsLoss().to(self.device)
        else:
            fn_loss = nn.BCEWithLogitsLoss().to(self.device)
        
        # Optimizer 
        optim = torch.optim.Adam(net.parameters(), lr=self.lr)
        
        # functions
        fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
        fn_denorm = lambda x, mean, std: (x * std) + mean
        fn_class = lambda x: 1.0 * (x > 0.5)

        
        # Set SummaryWriter for Tensorboard
        writer_train = SummaryWriter(log_dir=os.path.join(self.log_dir, 'train'))
        writer_val = SummaryWriter(log_dir=os.path.join(self.log_dir, 'val'))
        
        # Train network
        st_epoch = 0
    
        if self.train_continue == "on":
            net, optim, st_epoch = load(ckpt_dir=self.ckpt_dir, net=net, optim=optim)
        
        epochbar = tqdm(range(st_epoch + 1, self.num_epoch + 1),
                        total = len(range(st_epoch + 1, self.num_epoch + 1)),
                        desc = f'EPOCH {st_epoch}/{self.num_epoch}',
                        leave = True,
                        )
        for epoch in epochbar:
            epochbar.set_description(f'EPOCH {epoch}/{self.num_epoch}')
            net.train()
            loss_arr = []
            
            trainbar = tqdm(enumerate(loader_train, 1),
                            total = int(num_batch_train),
                            desc = f'TRAIN 0/{int(num_batch_train)}',
                            leave = False,
                            )
            for batch, data in trainbar:
                trainbar.set_description(f'TRAIN {batch}/{int(num_batch_train)}')
                # forward pass
                label = data['label'].to(self.device)
                label_bce = data['label_bce'].to(self.device)
                input = data['input'].to(self.device)
    
                output = net(input)
                
                #if not self.loss == 'BCE':
                    #output = sig(output)
                    #output = fn_class(output)
                    #output = output.squeeze()
                    #out_zero = torch.zeros(output.shape).to(self.device).float()
                    #output = torch.stack([output, out_zero], dim=1)
                    
                    
                    #if batch==5:
                    #    tf(output[0]).show()
    
                # backward pass
                optim.zero_grad()
            
                #new_output = fn_class(output)
                #new_output = new_output[:, 0:2, :, :]
                #new_output = Variable(new_output.data, requires_grad=True)
    
                loss = fn_loss(output, label_bce) if self.loss == 'BCE' else fn_loss(output, label.squeeze().long())
                loss.backward()
                
                #if not self.loss == 'BCE':
                    #output = sig(output)
                    #temp = fn_class(output)
                    #output[:,0,:,:] = torch.zeros((1, output.shape[2], output.shape[3]))
                    #output[:,2,:,:] = torch.zeros((1, output.shape[2], output.shape[3]))
                    #output = Variable(output.data, requires_grad=True)
                    
                    #if batch==5:
                        #tf(temp[0]).show()
    
                optim.step()
    
                # calculate loss
                loss_arr += [loss.item()]
                
                trainbar.set_postfix({"LOSS": f"{np.mean(loss_arr):.4f}"})
    
                #print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                #      (epoch, self.num_epoch, batch, num_batch_train, np.mean(loss_arr)))
                
    
                # Save to Tensorboard 
                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))
    
                #writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
                #writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
                #writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
    
            #writer_train.add_scalar('loss', np.mean(loss_arr), epoch)
            trainbar.close()
            
    
            with torch.no_grad():
                net.eval()
                loss_arr = []
                validbar = tqdm(enumerate(loader_val, 1),
                                total = int(num_batch_val),
                                desc = f'VALID 0/{int(num_batch_val)}',
                                leave = False,
                                )
    
                for batch, data in validbar:
                    validbar.set_description(f'VALID {batch}/{int(num_batch_val)}')
                    # forward pass
                    label = data['label'].to(self.device)
                    label_bce = data['label_bce'].to(self.device)
                    input = data['input'].to(self.device)
    
                    output = net(input)
                    #if not self.loss == 'BCE':
                        #output = sig(output)
                        #output = output.squeeze()
                        #out_zero = torch.zeros(output.shape).to(self.device).float()
                        #output = torch.stack([output, out_zero], dim=1)
                        #output[:,0,:,:] = torch.zeros((1, output.shape[2], output.shape[3]))
                        #output[:,2,:,:] = torch.zeros((1, output.shape[2], output.shape[3]))
                        #output = Variable(output.data, requires_grad=True)
                    #new_output = fn_class(output)
                    #new_output = new_output[:, 0:2, :, :]
                    #new_output = Variable(new_output.data, requires_grad=True)
                    
    
                    # calculate loss
                    loss = fn_loss(output, label_bce) if self.loss == 'BCE' else fn_loss(output, label.squeeze().long())
    
                    loss_arr += [loss.item()]
                                             
                    validbar.set_postfix({"LOSS": f"{np.mean(loss_arr):.4f}"})
    
                    #print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                    #      (epoch, self.num_epoch, batch, num_batch_val, np.mean(loss_arr)))
    
                    # Save to Tensorboard
                    label = fn_tonumpy(label)
                    input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                    #if not self.loss == 'BCE':
                    #    output = torch.cat([output, out_zero.unsqueeze(1)], dim=1)
                    output = fn_tonumpy(fn_class(sig(output)))# if self.loss=='BCE' else fn_tonumpy(fn_class(output)[:,1,:,:].unsqueeze(1))
    
                    #writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                    #writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                    #writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                    
                    #if batch==num_batch_val:
                    for j in range(label.shape[0]):
                        id = num_batch_val * (batch - 1) + j
        
                        plt.imsave(os.path.join(self.result_dir, 'png', '%04d_label.png' % id), label[j].squeeze(), cmap='gray')
                        plt.imsave(os.path.join(self.result_dir, 'png', '%04d_input.png' % id), input[j].squeeze(), cmap='gray')
                        plt.imsave(os.path.join(self.result_dir, 'png', '%04d_output.png' % id), output[j].squeeze(), cmap='gray')
                                             
            validbar.close()
    
            #writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
    
            if epoch % 5 == 0:
                save(ckpt_dir=self.ckpt_dir, net=net, optim=optim, epoch=epoch)
    
        epochbar.close()
        #writer_train.close()
        #writer_val.close()