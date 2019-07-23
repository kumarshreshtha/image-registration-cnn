from skimage import transform
import numpy as np
import torch


class IntensityNorm(object):
    def __init__(self, min_clip=0, max_clip=1300):
        self.min_clip = min_clip
        self.max_clip = max_clip

    def __call__(self, inputs):
        source, target = inputs
        source = np.clip(source, self.min_clip, self.max_clip)//self.max_clip
        target = np.clip(target, self.min_clip, self.max_clip)//self.max_clip
        return source, target


class Rescale(object):
    def __init__(self, factor=2/3):
        self.factor = factor

    def __call__(self, inputs):
        source, target = inputs
        source = transform.Rescale(source, self.factor, multichannel=True)
        target = transform.Rescale(target, self.factor, multichannel=True)
        return source, target


class ToTensor(object):
    def __call__(self, inputs):
        source, target = inputs
        source = source.transpose((0, 4, 1, 2, 3))
        target = target.transpose((0, 4, 1, 2, 3))
        return torch.from_numpy(source), torch.from_numpy(target)
