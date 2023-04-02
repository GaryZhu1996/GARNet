#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

import torch


class RecallLoss(torch.nn.Module):
    
    def __init__(self, epsilon=0):
        super(RecallLoss, self).__init__()
        print('Recall loss')
        self.epsilon = epsilon
    
    def forward(self, predicts, targets):
        axes = tuple(range(1, len(predicts.shape)))
        intersection = predicts * targets
        
        recall = torch.sum(intersection, axes) / (torch.sum(targets, axes) + self.epsilon)
        loss = 1 - torch.mean(recall)
        return loss
