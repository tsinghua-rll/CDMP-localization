#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : tools.py
# Creation Date : 19-04-2018
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import cv2
import numpy as np 


def place_obj(dst, src, mask, x_offset, y_offset):
    tmp = dst.copy()
    h, w = mask.shape[:2] 
    tmp[x_offset:x_offset+h, y_offset:y_offset+w, :] = src 
    target_mask = np.zeros_like(dst)[..., 0]
    target_mask[x_offset:x_offset+h, y_offset:y_offset+w] = mask 
    target_mask = target_mask.astype(np.bool)
    dst[target_mask] = tmp[target_mask]
    return dst 


if __name__ == '__main__':
    dst = cv2.imread('../background/0.png')
    src = cv2.imread('../clipped_obj/sugar/0.png')
    mask = src[..., -1]
    src = src[..., 0:3]
    dst = place_obj(dst, src, mask, 1000, 1000)
    cv2.imwrite('../output.png', dst)
