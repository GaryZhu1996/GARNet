#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>
#
# References:
# - https://github.com/hzxie/Pix2Vox

import torch


class ChannelPool(torch.nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1),
                          torch.mean(x, 1).unsqueeze(1)), dim=1)
    
    
class FusionModule(torch.nn.Module):
    def __init__(self):
        super(FusionModule, self).__init__()
        self.sa = torch.nn.Sequential(
            ChannelPool(),
            torch.nn.Conv3d(2, 1, kernel_size=5, padding=2),
            # torch.nn.BatchNorm3d(1),
            # torch.nn.ReLU()
        )
        self.ca_avg_pool = torch.nn.AdaptiveAvgPool3d(1)
        self.ca_max_pool = torch.nn.AdaptiveMaxPool3d(1)
        self.ca_conv = torch.nn.Conv1d(1, 1, kernel_size=9, padding=4, bias=False)

    def forward(self, features):
        sa_features = self.sa(features).expand_as(features)
        # avg_pooling
        avgout = \
            self.ca_conv(self.ca_avg_pool(features).squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        
        # max_pooling
        maxout = \
            self.ca_conv(self.ca_max_pool(features).squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        ca_features = (avgout + maxout).expand_as(features)
        return sa_features + ca_features


class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(1568, 512, kernel_size=4, stride=2,
                                     bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2,
                                     bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2,
                                     bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2,
                                     bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1,
                                     bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.Sigmoid()
        )
        self.fusion_module = FusionModule()

    def forward(self, image_features, freeze_fusion=False):
        views_num = image_features.shape[1]
        image_features = image_features.view(-1, 256, 7, 7)
        # torch.Size([batch_size * view_num, 256, 7, 7])
        image_features = image_features.view(-1, 1568, 2, 2, 2)
        # torch.Size([batch_size * view_num, 1568, 2, 2, 2])
        
        # fusion
        branch_features = self.layer1(image_features)
        # torch.Size([batch_size * view_num, 512, 4, 4, 4])
        if not freeze_fusion:
            feature_weights = self.fusion_module(branch_features).view(-1, views_num, 512, 4, 4, 4)
            # torch.Size([batch_size, view_num, 512, 4, 4, 4])
            feature_weights = torch.softmax(feature_weights, dim=1)
            branch_features = branch_features.view(-1, views_num, 512, 4, 4, 4)
            fusion_feature = branch_features * feature_weights
            fusion_feature = torch.sum(fusion_feature, dim=1).unsqueeze(dim=1)
            # torch.Size([batch_size, 1, 512, 4, 4, 4])
            branch_features = torch.cat([branch_features, fusion_feature], dim=1).view(-1, 512, 4, 4, 4)
            # torch.Size([batch_size * (view_num + 1), 512, 4, 4, 4])
            views_num += 1
        
        gen_volumes = self.layer2(branch_features)
        # torch.Size([batch_size * (view_num) or (view_num + 1), 128, 8, 8, 8])
        gen_volumes = self.layer3(gen_volumes)
        # torch.Size([batch_size * (view_num) or (view_num + 1), 32, 16, 16, 16])
        gen_volumes = self.layer4(gen_volumes)
        # torch.Size([batch_size * (view_num) or (view_num + 1), 8, 32, 32, 32])
        raw_features = gen_volumes.view(-1, views_num, 8, 32, 32, 32)
        # torch.Size([batch_size, (view_num) or (view_num + 1), 8, 32, 32, 32])
        gen_volumes = self.layer5(gen_volumes)
        # torch.Size([batch_size * (view_num) or (view_num + 1), 1, 32, 32, 32])
        gen_volumes = gen_volumes.view(-1, views_num, 1, 32, 32, 32)
        # torch.Size([batch_size, (view_num) or (view_num + 1), 1, 32, 32, 32])
        raw_features = torch.cat((raw_features, gen_volumes), dim=2)
        # torch.Size([batch_size, (view_num) or (view_num + 1), 9, 32, 32, 32])
        gen_volumes = torch.squeeze(gen_volumes, dim=2)
        # torch.Size([batch_size, (view_num) or (view_num + 1), 32, 32, 32])
        return raw_features, gen_volumes
