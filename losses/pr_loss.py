#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

import torch
from losses import precision_loss, recall_loss


class PRLoss(torch.nn.Module):
    
    def __init__(self, epsilon=0):
        super(PRLoss, self).__init__()
        print('Precision & Recall loss')
        self.precision_loss = precision_loss.PrecisionLoss(epsilon=epsilon)
        self.recall_loss = recall_loss.RecallLoss(epsilon=epsilon)
        self.epsilon = epsilon
    
    def forward(self, predicts, targets):
        loss = self.precision_loss(predicts, targets) + self.recall_loss(predicts, targets)
        return loss
