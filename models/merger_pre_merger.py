#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>
#
# References:
# - https://github.com/hzxie/Pix2Vox

import torch


class Merger(torch.nn.Module):
    def __init__(self, cfg):
        super(Merger, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(18, 9, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(9),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 9, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(9),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 9, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(9),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 9, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(9),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(36, 9, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(9),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 1, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        
    def calculate_score_map(self, views_num, raw_features):
        fusion_feature = raw_features[:, -1, ::].unsqueeze(dim=1).expand(-1, views_num, -1, -1, -1, -1)
        # torch.Size([batch_size, view_num, 9, 32, 32, 32])
        raw_features = torch.cat([raw_features[:, :-1, ::], fusion_feature - raw_features[:, :-1, ::]], dim=2)
        # torch.Size([batch_size, view_num , 18, 32, 32, 32])
        raw_features = raw_features.view(-1, 18, 32, 32, 32)
        # torch.Size([batch_size * view_num , 18, 32, 32, 32])
    
        volume_weights1 = self.layer1(raw_features)
        # torch.Size([batch_size * view_num, 9, 32, 32, 32])
        volume_weights2 = self.layer2(volume_weights1)
        # torch.Size([batch_size * view_num, 9, 32, 32, 32])
        volume_weights3 = self.layer3(volume_weights2)
        # torch.Size([batch_size * view_num, 9, 32, 32, 32])
        volume_weights4 = self.layer4(volume_weights3)
        # torch.Size([batch_size * view_num, 9, 32, 32, 32])
        volume_weights = self.layer5(torch.cat([
            volume_weights1, volume_weights2, volume_weights3, volume_weights4
        ], dim=1))
        # torch.Size([batch_size * view_num, 9, 32, 32, 32])
        volume_weights = self.layer6(volume_weights)
        # torch.Size([batch_size * view_num, 1, 32, 32, 32])
    
        volume_weights = volume_weights.view(-1, views_num, 1, 32, 32, 32).squeeze(dim=2)
        # torch.Size([batch_size, view_num, 32, 32, 32])
        volume_weights = torch.softmax(volume_weights, dim=1)
        # print(volume_weights.size())  # torch.Size([batch_size, n_views, 32, 32, 32])
        
        return volume_weights

    def forward(self, raw_features, coarse_volumes):
        views_num = coarse_volumes.shape[1] - 1
        volume_weights = self.calculate_score_map(views_num, raw_features)
        
        # print(coarse_volumes.size())  # torch.Size([batch_size, n_views, 32, 32, 32])
        coarse_volumes = coarse_volumes[:, :-1, ::] * volume_weights
        coarse_volumes = torch.sum(coarse_volumes, dim=1)

        return torch.clamp(coarse_volumes, min=0, max=1)
