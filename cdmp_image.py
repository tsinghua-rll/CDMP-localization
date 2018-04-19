#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : cdmp_image.py
# Purpose :
# Creation Date : 19-04-2018
# Last Modified : 2018年04月19日 星期四 19时29分14秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import torch 
import torch.utils.data as data 
import cv2
import numpy as np

from data.scripts.data import CDMP_Synthesis

def collect_fn_image(batch):
    pass


def collect_fn_image_localization(batch):
    pass


class CDMP_Image(data.Dataset):
    def __init__(self, data_path, dataset_size, image_size=224, obj_sum=10, 
            collision_radius=600, border=300):
        self.data = CDMP_Synthesis(data_path, obj_sum, collision_radius, border) 
        self.dataset_size = dataset_size 
        self.label_dict = self.data.get_all_labels()

    def __getitem__(self, index):
        # (C, H, W)
        # trajs 
        pass

    def __len__(self):
        return self.dataset_size 


class CDMP_Image_Localization(data.Dataset):
    def __init__(self, data_path, dataset_size, image_size=224, obj_size=224, obj_sum=10, 
            collision_radius=600, border=300):
        self.data = CDMP_Synthesis(data_path, obj_sum, collision_radius, border) 
        self.dataset_size = dataset_size 
        self.label_dict = self.data.get_all_labels()

    def __getitem__(self, index):
        # img (C, H, W)
        # object_img (C, H, W)
        # gt ([x, y, id])
        bg, ret = 
        

    def __len__(self):
        return self.dataset_size 



if __name__ == '__main__':
    pass	
