#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

import torch


class DownSampling(torch.nn.Module):
    def __init__(self, channels):
        super(DownSampling, self).__init__()
        self.pooling = torch.nn.MaxPool3d(2)
        self.residual_block = torch.nn.Sequential(
            torch.nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm3d(channels),
            torch.nn.ReLU(),
            torch.nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm3d(channels),
        )
        self.activation_function = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.pooling(x)
        return self.activation_function(self.residual_block(x) + x)


class UpSampling(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampling, self).__init__()
        self.upsampling_module = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.upsampling_module(x)
        return x  # 64


class Refiner(torch.nn.Module):
    def __init__(self, cfg):
        super(Refiner, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 64, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),
        )
        
        self.down_block1 = DownSampling(64)
        self.down_block2 = DownSampling(64)
        self.down_block3 = DownSampling(64)
        
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),
        )
        
        self.up_block1 = UpSampling(64, 64)
        self.up_block2 = UpSampling(64, 64)
        self.up_block3 = UpSampling(64, 64)
        
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
            torch.nn.Conv3d(32, 1, kernel_size=3, padding=1, bias=False),
            torch.nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = x.unsqueeze(dim=1)  # torch.Size([batch_size, 1, 32, 32, 32])
        x = self.conv1(x)  # torch.Size([batch_size, 64, 32, 32, 32])
        x1 = self.down_block1(x)  # torch.Size([batch_size, 64, 16, 16, 16])
        x2 = self.down_block2(x1)  # torch.Size([batch_size, 64, 8 8, 8])
        x3 = self.down_block3(x2)  # torch.Size([batch_size, 64, 4, 4, 4])
        x4 = self.conv2(x3)  # torch.Size([batch_size, 64, 4, 4, 4])
        x5 = self.conv3(x4)  # torch.Size([batch_size, 64, 4, 4, 4])

        out = self.up_block1(x3 + x4 + x5)  # torch.Size([batch_size, 64, 8, 8, 8])

        out = self.up_block2(out + x2)  # torch.Size([batch_size, 64, 16, 16, 16]

        out = self.up_block3(out + x1)  # torch.Size([batch_size, 64, 32, 32, 32]

        out = self.conv4(out + x)  # torch.Size([batch_size, 1, 32, 32, 32]
        
        return out.squeeze(dim=1)
