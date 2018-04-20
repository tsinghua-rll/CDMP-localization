#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : random_place.py
# Creation Date : 19-04-2018
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import cv2
import glob
import numpy as np
import os

from .tools import place_obj
from bridson import poisson_disc_samples


class CDMP_Synthesis(object):
    def __init__(self, data_path, obj_sum=10, collision_radius=50, border=25, image_size=224, object_size=48):
        self.data_path = data_path
        self.obj_sum = obj_sum
        self.collision_radius = collision_radius
        self.border = border
        self.cls_dict, self.label = self.get_all_labels()
        self.bg_h = 240
        self.bg_w = 270
        self.image_size = image_size 
        self.object_size = object_size

    def get_all_labels(self):
        obj_cls = [i.split(
            '/')[-1] for i in glob.glob(os.path.join(self.data_path, 'clipped_obj/*'))]
        obj_cls.sort()
        res = {}
        for ind, obj in enumerate(obj_cls):
            res[obj] = ind
        return res, obj_cls

    def random_place(self, obj=False, seed=0):
        np.random.seed(seed)
        obj_cls = np.array([i.split(
            '/')[-1] for i in glob.glob(os.path.join(self.data_path, 'clipped_obj/*'))])
        obj_f = []
        # choose obj
        ind = np.arange(0, len(obj_cls))
        np.random.shuffle(ind)
        ind = ind[:self.obj_sum]
        obj_cls = obj_cls[ind]

        for cls in obj_cls:
            obj_f.append((cls, glob.glob(os.path.join(
                self.data_path, 'clipped_obj/{}/*.png'.format(cls)))))

        # choose obj pose
        pose_id = np.random.choice(len(obj_f[0][1]), self.obj_sum)
        f = []
        for ind, (cls, fl) in enumerate(obj_f):
            f.append(fl[pose_id[ind]])

        obj_img = []
        for name in f:
            im = cv2.imread(name, cv2.IMREAD_UNCHANGED)
            im = cv2.resize(im, (int(im.shape[1]), int(im.shape[0])))
            obj_img.append(im)

        redo = True
        while redo:
            bg = cv2.imread(os.path.join(self.data_path, 'background/0.png'))
            bg = cv2.resize(bg, (self.image_size, self.image_size))
            h, w = bg.shape[:2]
            while True:
                res = np.array(poisson_disc_samples(
                    h-self.border, w-self.border, self.collision_radius, random=np.random.rand))
                if res.shape[0] >= self.obj_sum:
                    ind = np.arange(res.shape[0])
                    np.random.shuffle(ind)
                    ind = ind[:self.obj_sum]
                    res = res[ind, :].astype(np.int32)
                    break

            ret = []
            for img, cls, coord in zip(obj_img, obj_cls, res):
                try:
                    # print(img.shape, coord)
                    bg = place_obj(bg, img[..., 0:3],
                                   img[..., -1], coord[0], coord[1])
                    ret.append([coord[0]+img.shape[0]//2, coord[1] +
                                img.shape[1]//2, self.cls_dict[cls]])
                except:
                    break
            else:
                redo = False

        # (H, W, C) [(H, W, C)], [[x,y,id]]
        if obj:
            obj_img = [cv2.resize(cv2.imread(name), (self.object_size, self.object_size)) for name in f]
            return bg, obj_img, ret
        else:
            return bg, ret


if __name__ == '__main__':
    data = CDMP_Synthesis('..', 5)
    bg, ret = data.random_place()
    print(data.get_all_labels())
    print(ret)
    bg = cv2.circle(bg, (ret[0][1], ret[0][0]), 10, (255, 0, 0), 10)
    cv2.imwrite('output.png', bg)
