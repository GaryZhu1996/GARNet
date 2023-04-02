#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>
#
# References:
# - https://github.com/hzxie/Pix2Vox

import torch
import torchvision.models


class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet = torch.nn.Sequential(*[
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3,
            resnet.layer4
        ])[:6]
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

    def init_backbone(self):
        resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet = torch.nn.Sequential(*[
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3,
            resnet.layer4
        ])[:6]
        
    def resnet_forward(self, image):
        image = image.view(-1, 3, 224, 224)
        
        features = self.resnet[0](image)
        features = self.resnet[1](features)
        features = self.resnet[2](features)
        features = self.resnet[3](features)
        features = self.resnet[4](features)
        # torch.Size([batch_size * view_num, 256, 56, 56])
        return features
    
    def other_layers_forward(self, image_features, views_num):
        image_features = self.resnet[5](image_features)
        image_features = self.layer1(image_features)
        # torch.Size([batch_size * view_num, 512, 28, 28])
        image_features = self.layer2(image_features)
        # torch.Size([batch_size * view_num, 256, 14, 14])
        image_features = self.layer3(image_features)
        # torch.Size([batch_size * view_num, 256, 7, 7])
        image_features = image_features.view(-1, views_num, 256, 7, 7)
        return image_features
        
    def forward(self, rendering_images):
        views_num = rendering_images.shape[1]
        # rendering_images = rendering_images.view(-1, 3, 224, 224)
        # image_features = self.resnet(rendering_images)
        image_features = self.resnet_forward(rendering_images)
        # torch.Size([batch_size * view_num, 512, 28, 28])
        image_features = self.other_layers_forward(image_features, views_num)
        # torch.Size([batch_size, view_num, 256, 7, 7])
        return image_features
