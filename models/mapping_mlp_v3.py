#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

import torch


class MLP(torch.nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()
        if len(cfg.REDUCE.MLP_DIM) == 3:
            self.avg_pool = torch.nn.AdaptiveAvgPool2d(7)
            self.max_pool = torch.nn.AdaptiveMaxPool2d(7)
            self.mlp_1 = torch.nn.Sequential(
                torch.nn.Linear(256 * 7 * 7, cfg.REDUCE.MLP_DIM[0]),
                torch.nn.ReLU(),
                torch.nn.Linear(cfg.REDUCE.MLP_DIM[0], cfg.REDUCE.MLP_DIM[1]),
                torch.nn.ReLU()
            )
            self.mlp_2 = torch.nn.Linear(cfg.REDUCE.MLP_DIM[1], cfg.REDUCE.MLP_DIM[2])

    def forward(self, x):
        # x = self.mlp_1(self.avg_pool(x).squeeze()) + self.mlp_1(self.max_pool(x).squeeze())
        # x = self.mlp_1(self.max_pool(x).squeeze())
        x = x.view(-1, 256, 56, 56)
        x = self.mlp_1(self.avg_pool(x).view(-1, 7 * 7 * 256)) + self.mlp_1(self.max_pool(x).view(-1, 7 * 7 * 256))
        return self.mlp_2(x)
