import os
import torch
import numpy as np
import random
import scipy 
import skimage.transform

from torchvision import transforms

from torch.utils.data.sampler import SubsetRandomSampler

import warnings
warnings.filterwarnings("ignore")

class Sin2t(object):
	def __call__(self, sample):
		x, label = sample['x'], sample['label']
		x2 = np.zeros(x.shape[:-3] + (1,) + x.shape[-2:])
		x2[..., 0, :, :] = 2 * x[..., 0, :, :] * x[..., 1, :, :]
		del x
		
		return {'x': x2, 'label': label}
		
class Cos2t(object):
	def __call__(self, sample):
		x, label = sample['x'], sample['label']
		x2 = np.zeros(x.shape[:-3] + (1,) + x.shape[-2:])
		x2[..., 0, :, :] = x[..., 0, :, :]**2 + x[..., 1, :, :]**2
		del x
		
		return {'x': x2, 'label': label}
		
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
		x, label = sample['x'], sample['label']
		
		dims = x.shape[-self.ndims:]
		corner = np.array([np.random.randint(0, d-nd) for d, nd in zip(dims, self.crop_size)])

		crop_indices = tuple(np.s_[c:c+nd] for c, nd in zip(corner, self.crop_size))
		ds_indices = tuple(np.s_[0:d] for d in x.shape[:-self.ndims])

		d_indices = ds_indices + crop_indices
		
		return {'x':x[d_indices], 'label':label}


class RandomTranspose(object):
	def __init__(self, prob=0.5, ndims=2):
		self.prob  = prob
		self.ndims = ndims

	def __call__(self, sample):
		x, label = sample['x'], sample['label']
		
		if np.random.random() < self.prob:
			axes = random.sample(range(-self.ndims, 0), 2)
			x = np.swapaxes(x, axes[0], axes[1])
			
		return	{'x':x, 'label':label}
	
	
class RandomFlip(object):
	def __init__(self, prob=0.5, ndims=2):
		self.prob  = prob
		self.ndims = ndims

	def __call__(self, sample):
		x, label = sample['x'], sample['label']
		
		for dim in range(-self.ndims, 0):
			if np.random.random() < self.prob:
				x = np.flip(x, axis=dim)
				
		return	{'x':x, 'label':label}

class RandomShift(object):
	def __init__(self, frac=0.5, ndims=2):
		self.frac = frac
		self.ndims = ndims
		
	def __call__(self, sample):
		x, label = sample['x'], sample['label']
		
		for i, d in enumerate(x.shape[-self.ndims:]):
			toshift = np.random.randint(-int(self.frac * d), int(self.frac * d))
			x =	np.roll(x, toshift, axis=-self.ndims+i)
		return	{'x':x, 'label':label}
	
class RandomRotation(object):
	def __init__(self, degree, ndims=2):
		self.degree = degree
		self.ndims	= ndims

	def __call__(self, sample):
		x, label = sample['x'], sample['label']
		
		deg = np.random.uniform(-self.degree, self.degree)
		axes = random.sample(range(-self.ndims, 0), 2)
		x = scipy.ndimage.rotate(x, deg, axes=(axes[0], axes[1]), mode='wrap', order=1)
		
		return	{'x':x, 'label':label}

class AverageTimeLabel(object):
	def __call__(self, sample):
		x, label = sample['x'], sample['label']
		return	{'x': x, 'label': np.average(label, axis=-2)}

class SwapTimeChannelAxes(object):
	def __call__(self, sample):
		x, label = sample['x'], sample['label']
		return {'x': np.swapaxes(x, 0, 1), 'label': label}

class ToTensor(object):
	def __call__(self, sample):
		x, label = sample['x'], sample['label']

		return {'x': torch.FloatTensor(x.copy()), 
				'label':torch.FloatTensor(label.copy())}
