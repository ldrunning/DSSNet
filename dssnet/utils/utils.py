import os
import torch
import numpy as np
import cv2
from skimage.color import separate_stains, combine_stains, rgb2hed, hed2rgb


def tonumpy(image):
    return np.array(image)


def random_channel(image):
    channel_list = np.arange(3)
    np.random.shuffle(channel_list)
    k = np.random.uniform(0, 1)
    if 0.8 < k:
        return image.copy()
    else:
        return image[:, :, channel_list].copy()


def random_blur(image):
    blur_type = np.random.randint(0, 8)
    ksize = np.random.randint(0, 2) * 2 + 1
    if blur_type == 0:
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif blur_type == 1:
        # 均值 blur
        return cv2.blur(image, (ksize, ksize))
    elif blur_type == 2:
        return cv2.medianBlur(image, ksize)
    # elif blur_type == 3:
    #     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    #     return cv2.filter2D(image, -1, kernel=kernel)
    else:
        return image


class ColorJitterHED:
    def __init__(self, h, e, d):
        self.h = h
        self.e = e
        self.d = d

    def __call__(self, image):
        image = np.array(image)
        hed = rgb2hed(image)
        h_t = np.random.uniform(1 - self.h, 1 + self.h)
        e_t = np.random.uniform(1 - self.e, 1 + self.e)
        d_t = np.random.uniform(1 - self.d, 1 + self.d)
        hed[:, :, 0] *= h_t
        hed[:, :, 1] *= e_t
        hed[:, :, 2] *= d_t
        image = hed2rgb(hed)
        image = (np.minimum(np.maximum(image, 0), 1) * 255).astype(np.uint8)
        return image


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ColorJitterHRD:
    def __init__(self, h, r, d):
        self.h = h
        self.r = r
        self.d = d
        self.rgb_from_hrd = np.array([[0.644, 0.710, 0.285],
                                      [0.0326, 0.873, 0.487],
                                      [0.270, 0.562, 0.781]])
        self.hrd_from_rgb = np.linalg.inv(self.rgb_from_hrd)

    def __call__(self, image):
        image = np.array(image)
        hrd = separate_stains(image, self.hrd_from_rgb)
        h_t = np.random.uniform(1 - self.h, 1 + self.h)
        r_t = np.random.uniform(1 - self.r, 1 + self.r)
        d_t = np.random.uniform(1 - self.d, 1 + self.d)
        hrd[:, :, 0] *= h_t
        hrd[:, :, 1] *= r_t
        hrd[:, :, 2] *= d_t
        image = combine_stains(hrd, self.rgb_from_hrd)
        image = (np.minimum(np.maximum(image, 0), 1) * 255).astype(np.uint8)
        return image
