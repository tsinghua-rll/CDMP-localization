#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : model.py
# Purpose :
# Creation Date : 19-04-2018
# Last Modified : 2018年04月19日 星期四 14时22分51秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models


class CDMP_Localization(nn.Module):
    def __init__(self, input_size, object_size):
        super(CDMP_Localization, self).__init__()
        self.input_size = input_size 
        self.object_size = object_size 

        self.resnet_img = models.resnet18(pretrained=False, num_classes=256)
        self.resnet_obj = models.resnet18(pretrained=False, num_classes=64)
        param = torch.load('./assets/resnet18.pth')
        del param['fc.weight']
        del param['fc.bias']
        self.resnet_img.load_state_dict(param, strict=False)
        self.resnet_obj.load_state_dict(param, strict=False)
        self.fc1 = nn.Linear(256+64, 256)
        self.center = nn.Linear(256, 2)

    def forward(self, img, obj_img):
        img_x = self.resnet_img(img)
        obj_img_x = self.resnet_obj(obj_img)

        x = F.relu(self.fc1(torch.cat([img_x, obj_img_x], -1)))
        
        return self.center(x)


if __name__ == '__main__':
    pass	
