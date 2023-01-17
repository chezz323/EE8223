import os
from glob import glob
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        #lst_data = os.listdir(self.data_dir)

        #lst_label = [f for f in lst_data if f.startswith('label')]
        #lst_input = [f for f in lst_data if f.startswith('input')]
        
        # data_dir: '../data/brain_mri'
        lst_label = data_dir['mask'].tolist()
        lst_label = [path.replace("\\", "/") for path in lst_label]
        lst_input = data_dir['images'].tolist()
        lst_input = [path.replace("\\", "/") for path in lst_input]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        #label = np.load(self.lst_label[index])
        #input = np.load(self.lst_input[index])
        label = cv2.imread(self.lst_label[index], cv2.IMREAD_GRAYSCALE)#, 0)
        label_bce = cv2.imread(self.lst_label[index])
        input = cv2.imread(self.lst_input[index])#, 0)

        label = label/255.0
        label_bce = label_bce/255.0
        input = input/255.0

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label, 'label_bce': label_bce}

        if self.transform:
            data = self.transform(data)

        return data


## Trainsform
class ToTensor(object):
    def __call__(self, data):
        label, input, label_bce = data['label'], data['input'], data['label_bce']

        label = label.transpose((2, 0, 1)).astype(np.float32)
        label_bce = label_bce.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input), 'label_bce': torch.from_numpy(label_bce)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input, label_bce = data['label'], data['input'], data['label_bce']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input, 'label_bce': label_bce}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input, label_bce = data['label'], data['input'], data['label_bce']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            label_bce = np.fliplr(label_bce)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            label_bce = np.flipud(label_bce)
            input = np.flipud(input)

        data = {'label': label, 'input': input, 'label_bce': label_bce}

        return data

