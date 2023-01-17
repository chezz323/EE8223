import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
###############################################################################
# Dice Loss
###############################################################################
import torch.nn.functional as F
from typing import Optional
from one_hot import one_hot

from torchmetrics.classification import BinaryF1Score

class DiceCoefficientScore(nn.Module):
    """
    Shape:
        - Input: :math:`(N, 1, H, W)` 
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    """

    def __init__(self) -> None:
        super(DiceCoefficientScore, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        #input_soft = F.softmax(input, dim=1)
        
        input_sig = input.squeeze()
        if input_sig.ndim==2:
            input_sig = input_sig.unsqueeze(0)
        zero = torch.zeros(input_sig.shape).to(input.device).float()
        input_sig = torch.stack([zero, input_sig], dim=1)
        input_sig = Variable(input_sig.data, requires_grad=True)
        
        #input_sig = one_hot(input_sig, num_classes=2,
        #                    device=input.device, dtype=input.dtype)
        #F.softmax(input_sig, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=2,
                                 device=input.device, dtype=input.dtype)
        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_sig * target_one_hot, dims)
        cardinality = torch.sum(input_sig + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(dice_score)
    
class F1Score(nn.Module):
    def __int__(self):
        super(F1Score, self).__init__()
        
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        f1score = BinaryF1Score().to(input.device)
        input = input.squeeze()
        if input.ndim==2:
            input = input.unsqueeze(0)
        input = torch.reshape(input, (-1,))
        target = torch.reshape(target.squeeze().long(), (-1,))
        return f1score(input, target)
                