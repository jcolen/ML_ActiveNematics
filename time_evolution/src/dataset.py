import torch
from torch.utils.data import Dataset
import numpy as np
import random

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SinCos(object):
    def __call__(self, image):
        image2 = np.zeros(image.shape)
        image2[..., 0, :, :] = 2 * image[..., 0, :, :] * image[..., 1, :, :]
        image2[..., 1, :, :] = image[..., 0, :, :]**2 - image[..., 1, :, :]**2
        del image

        return image2
    
class RandomFlip(object):
    def __init__(self, prob=0.5, ndims=2):
        self.prob  = prob
        self.ndims = ndims

    def __call__(self, image):        
        for dim in range(-self.ndims, 0):
            if np.random.random() < self.prob:
                image = np.flip(image, axis=dim)

        return image

class RandomTranspose(object):
    def __init__(self, prob=0.5, ndims=2):
        self.prob  = prob
        self.ndims = ndims

    def __call__(self, image):
        if np.random.random() < self.prob:
            axes = random.sample(range(-self.ndims, 0), 2)
            image = np.swapaxes(image, axes[0], axes[1])
            
        return image

class RandomShift(object):
    def __init__(self, frac=0.5, ndims=2):
        self.frac = frac
        self.ndims = ndims
        
    def __call__(self, image):        
        for i, d in enumerate(image.shape[-self.ndims:]):
            toshift = np.random.randint(-int(self.frac * d), int(self.frac * d))
            image =  np.roll(image, toshift, axis=-self.ndims+i)
        return  image

class RandomCrop(object):
    def __init__(self, crop_size=48):
        self.crop_size = crop_size

    def __call__(self, image):        
        cy = np.random.randint(0, image.shape[-2]-self.crop_size)
        cx = np.random.randint(0, image.shape[-1]-self.crop_size)
        image = image[..., cy:cy+self.crop_size, cx:cx+self.crop_size]

        return image

class ToTensor(object):
    def __call__(self, image):
        return torch.tensor(image, dtype=torch.float32)

import h5py
class NematicsSequenceDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 transform=None,
                 frames_per_seq=8):
        super().__init__()
        self.path = path
        self.transform = transform
        self.frames_per_seq = frames_per_seq

        self.dataset = None
        logger.info(f'Found director field at {path} with {len(self)} sequences')

    def __len__(self):
        if self.dataset is None:
            with h5py.File(self.path, 'r') as h5f:
                return h5f['director'].shape[0] - self.frames_per_seq
        else:
            return self.dataset['director'].shape[0] - self.frames_per_seq

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.path, 'r')

        image = self.dataset['director'][idx:idx+self.frames_per_seq][()]
        if self.transform:
            image = self.transform(image)
        return image

