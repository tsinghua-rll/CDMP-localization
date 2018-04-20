#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : cdmp_image.py
# Purpose :
# Creation Date : 19-04-2018
# Last Modified : 2018年04月20日 星期五 16时57分15秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import torch 
import torch.utils.data as data 
import cv2
import numpy as np

from data.scripts.data import CDMP_Synthesis

def collect_fn_image(batch):
    pass


def collect_fn_image_localization(batch):
    img = []
    object_img = []
    target = []
    for sample in batch:
        img.append(sample[0])
        object_img.append(sample[1])
        target.append(torch.FloatTensor(sample[2]))
    return torch.stack(img, 0), torch.stack(object_img, 0), torch.stack(target, 0)


class CDMP_Image(data.Dataset):
    def __init__(self, data_path, dataset_size, image_size=224, obj_sum=10, 
            collision_radius=600, border=300):
        self.data = CDMP_Synthesis(data_path, obj_sum, collision_radius, border) 
        self.dataset_size = dataset_size 
        _, self.label = self.data.get_all_labels()
        self.image_size = image_size 
        self.obj_size = obj_size

    def __getitem__(self, index):
        # (C, H, W)
        # trajs 
        pass

    def __len__(self):
        return self.dataset_size 


class CDMP_Image_Localization(data.Dataset):
    def __init__(self, data_path, dataset_size, image_size=224, obj_size=48, obj_sum=10, 
            collision_radius=50, border=25):
        self.data = CDMP_Synthesis(data_path, obj_sum, collision_radius, border, image_size, obj_size) 
        self.dataset_size = dataset_size 
        _, self.label = self.data.get_all_labels()
        self.image_size = image_size 
        self.obj_size = obj_size

    def __getitem__(self, index):
        # img (C, H, W)
        # object_img (C, H, W)
        # gt ([x, y, id])
        img, obj_img, ret = self.data.random_place(obj=True, seed=index) 
        h, w = img.shape[:2]
        
        # pick an object
        ind = int(np.random.choice(len(ret), 1))
        obj_img = obj_img[ind][..., :3]
        x, y, label_id = ret[ind]
        
        # resize and normalized img 
        img = img/255
        # resize and normalized object 
        obj_img = obj_img/255

        # normalized coord 
        x /= h
        y /= w

        return torch.from_numpy(img).permute(2,0,1).float(), torch.from_numpy(obj_img).permute(2,0,1).float(), [x, y, label_id]


    def __len__(self):
        return self.dataset_size 



if __name__ == '__main__':
    pass	
