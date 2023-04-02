#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

import torch


class PrecisionLoss(torch.nn.Module):
    
    def __init__(self, epsilon=0):
        super(PrecisionLoss, self).__init__()
        print('Precision loss')
        self.epsilon = epsilon
    
    def forward(self, predicts, targets):
        axes = tuple(range(1, len(predicts.shape)))
        intersection = predicts * targets
        
        precision = torch.sum(intersection, axes) / (torch.sum(predicts, axes) + self.epsilon)
        loss = 1 - torch.mean(precision)
        return loss
