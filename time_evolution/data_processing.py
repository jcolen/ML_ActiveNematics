import os
import torch
import numpy as np
import random
import scipy 
import skimage.transform

from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")

class SinCos(object):
	def __call__(self, sample):
		image, label = sample['image'], sample['label']
		image2 = np.zeros(image.shape)
		image2[..., 0, :, :] = 2 * image[..., 0, :, :] * image[..., 1, :, :]
		image2[..., 1, :, :] = image[..., 0, :, :]**2 - image[..., 1, :, :]**2
		del image
		
		return {'image': image2, 'label': label}
    
class RandomCrop(object):
    def __init__(self, crop_size, ndims=2):
        self.ndims = ndims
        
        assert isinstance(crop_size, (int, tuple))
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, ) * self.ndims
        else:
            assert len(crop_size) == self.ndims
            self.crop_size = crop_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        dims = image.shape[-self.ndims:]
        corner = np.array([np.random.randint(0, d-nd) for d, nd in zip(dims, self.crop_size)])

        crop_indices = tuple(np.s_[c:c+nd] for c, nd in zip(corner, self.crop_size))
        same_indices = tuple(np.s_[0:d] for d in image.shape[:-self.ndims])

        indices = same_indices + crop_indices
        
        return {'image':image[indices], 'label':label}


    
    
class RandomTranspose(object):
    def __init__(self, prob=0.5, ndims=2):
        self.prob  = prob
        self.ndims = ndims

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        if np.random.random() < self.prob:
            axes = random.sample(range(-self.ndims, 0), 2)
            image = np.swapaxes(image, axes[0], axes[1])
            
        return {'image':image, 'label':label}
    

    
    
class RandomFlip(object):
    def __init__(self, prob=0.5, ndims=2):
        self.prob  = prob
        self.ndims = ndims

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        for dim in range(-self.ndims, 0):
            if np.random.random() < self.prob:
                image = np.flip(image, axis=dim)
                
        return {'image':image, 'label':label}


    

class RandomShift(object):
    def __init__(self, frac=0.5, ndims=2):
        self.frac = frac
        self.ndims = ndims
        
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        for i, d in enumerate(image.shape[-self.ndims:]):
            #print(d)
            toshift = np.random.randint(-int(self.frac * d), int(self.frac * d))
            image =  np.roll(image, toshift, axis=-self.ndims+i)
        return  {'image':image, 'label':label}
    


    
class RandomRotation(object):
    def __init__(self, degree, ndims=2):
        self.degree = degree
        self.ndims  = ndims

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        deg = np.random.uniform(-self.degree, self.degree)
        axes = random.sample(range(-self.ndims, 0), 2)
        image = scipy.ndimage.rotate(image, deg, axes=(axes[0], axes[1]), mode='wrap', order=1)
        
        return {'image':image, 'label':label}


    

class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        return {'image':torch.tensor(image, dtype=torch.float32), 'label':torch.tensor(label, dtype=torch.float32)}

