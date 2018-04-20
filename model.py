#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : model.py
# Purpose :
# Creation Date : 19-04-2018
# Last Modified : 2018年04月20日 星期五 17时07分38秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.nn.parameter import Parameter
import numpy as np


class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1)*temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        if self.data_format == 'NHWC':
            feature = feature.permute(0,2,3,1).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width)
        
        softmax_attention = F.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(Variable(self.pos_x)*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(Variable(self.pos_y)*softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return feature_keypoints


class CDMP_Localization(nn.Module):
    def __init__(self, input_size, object_size, channel=3):
        super(CDMP_Localization, self).__init__()
        self.input_size = input_size 
        self.object_size = object_size 
        self.ch = channel
        
        self.pool = torch.nn.MaxPool2d(2, padding=1, stride=2)
        # for image input
        self.conv1_img = torch.nn.Conv2d(self.ch, 64, kernel_size=4, padding=1, stride=2)
        self.conv2_img = torch.nn.Conv2d(64, 64, kernel_size=4, padding=1, stride=2)
        self.conv3_img = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_img = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5_img = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6_img = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.spatial_softmax = SpatialSoftmax(self.input_size // 2 // 2, self.input_size // 2 // 2, 64) # (N, 64*2)

        # for object input
        self.conv1_obj = torch.nn.Conv2d(self.ch, 64, kernel_size=3, padding=1)
        self.conv2_obj = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_obj = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # self.center = torch.nn.Linear(64*2 + 64 * (self.object_size // 2 // 2 // 2 // 2 // 2)**2 + 
        #         64 * (self.input_size // 2 // 2 // 2)**2, 2)
        self.center = torch.nn.Linear(128+53824+3136, 2)

    def forward(self, img, obj_img):
        batch_size = img.shape[0]
        img_x = F.relu(self.conv1_img(img))
        img_x = F.relu(self.conv2_img(img_x))
        img_x = F.relu(self.conv3_img(img_x))
        img_x = F.relu(self.conv4_img(img_x))
        img_x = F.relu(self.conv5_img(img_x))
        img_x = F.relu(self.conv6_img(img_x))
        points = self.spatial_softmax(img_x)
        feature = self.pool(img_x).view(batch_size, -1)

        obj_x = self.pool(F.relu(self.conv1_obj(obj_img)))
        obj_x = self.pool(F.relu(self.conv2_obj(obj_x)))
        obj_x = self.pool(F.relu(self.conv3_obj(obj_x))).view(batch_size, -1)
        # print(points.shape, feature.shape, obj_x.shape)

        return self.center(torch.cat([feature, obj_x, points], -1))


if __name__ == '__main__':
    pass	
