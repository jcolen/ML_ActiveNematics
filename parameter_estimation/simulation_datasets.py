import os
import re
import glob
import torch
import pandas as pd
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import Dataset
try:
	from torch.utils.data import SubsetRandomSampler
except:
	from torch.utils.data.sampler import SubsetRandomSampler

import warnings
warnings.filterwarnings("ignore")

def get_folder_info(path):
	fname = os.path.basename(os.path.normpath(path)).lower()

	#Tokenize path around numbers and remove delimiters
	toks = fname.split('_')
	label = {}
	curr, value = None, None
	for tok in toks:
		if tok[0] == 'z' and len(tok) > 1 and tok[1].isdigit():
			curr = 'z'
			tok = tok[1:]
		try:
			value = float(tok)
			label[curr] = value
			continue
		except:
			curr = tok
	if len(label.keys()) == 0:
		return None
	return label

'''
Dataset to hold active nematics information
	Custom datasets should overload things like:
		list_file_indices
		get_image
		get_label
		__getitem__ (for sequence datasets)
		__len__ (for sequence datasets)
'''
class NematicsDataset(Dataset):
	def __init__(self, 
				 root_dir, 
				 name=None,
				 transform=None,
				 label_info=get_folder_info,
				 validation_split=0.2,
				 force_load=False):
		self.root_dir = root_dir
		self.transform = transform
		self.label_info = label_info

		self.files_index_name = os.path.join(root_dir, 'index_simulation.csv')
		self.build_files_index(force_load)
		self.folders = self.dataframe.folder.unique()
		self.label_names = self.dataframe.columns.to_list()
		self.label_names.remove('folder')
		self.label_names.remove('idx')
		self.label_names.sort()
		self.dataframe.to_csv(self.files_index_name, index=False)
		self.num_folders = len(self.folders)
		print('Found %d files in %d folders' % (len(self.dataframe), self.num_folders))
	
		self.split_indices(validation_split)

	def split_indices(self, validation_split):
		split	= int(np.floor(validation_split * len(self)))
		indices = np.arange(len(self))
		np.random.shuffle(indices)
		self.train_indices, self.test_indices = indices[split:], indices[:split]

	def __len__(self):
		return len(self.dataframe)

	def list_file_indices(self, path):
		fnames = glob.glob(os.path.join(path, 'nx*'))
		inds = [list(map(int, re.findall(r'\d+', os.path.basename(fname))))[-1] for fname in fnames]
		return np.sort(inds).tolist()

	def build_files_index(self, force_load=False):
		if not force_load and os.path.exists(self.files_index_name):
			self.dataframe = pd.read_csv(self.files_index_name)
			return

		#Build an index of the images in different class subfolders
		folders, labels, idxs = [], {},[]
		for subdir in os.listdir(self.root_dir):
			dirpath = os.path.join(self.root_dir, subdir)
			if not os.path.isdir(dirpath):
				continue
			
			inds = self.list_file_indices(dirpath)
			if inds is None:
				continue
			
			label = self.label_info(dirpath)
			print('%s: Label = %s' % (dirpath, str(label)))
			if label is None:
				print('No label found for folder %s' % subdir)
				continue

			nimages = len(inds)
			folders += [subdir] * nimages
			idxs += inds
			for key in label:
				if key in labels:
					labels[key] += [label[key]] * nimages
				else:
					labels[key] = [label[key]] * nimages

		self.dataframe = pd.DataFrame(dict({'folder': folders, 'idx': idxs}, **labels))
	
	def get_loader(self, indices, batch_size, num_workers, pin_memory=True):
		sampler = SubsetRandomSampler(indices)
		loader = torch.utils.data.DataLoader(self, 
			batch_size=batch_size,
			num_workers=num_workers,
			sampler=sampler,
			pin_memory=pin_memory)
		return loader

	def get_image(self, idx):
		subdir = os.path.join(self.root_dir, self.dataframe.folder[idx])
		ind = self.dataframe.idx[idx]
		x = np.array([
			np.loadtxt(os.path.join(subdir, 'nx%d' % ind)),
			np.loadtxt(os.path.join(subdir, 'ny%d' % ind))])
		return x
	
	def get_label(self, idx):
		label = np.array([self.dataframe[key][idx] for key in self.label_names])
		return label

	def get_average_labels(self):
		return np.array([np.average(self.dataframe[key]) for key in self.label_names])

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		image = self.get_image(idx)
		label = self.get_label(idx)

		sample = {'x': image, 'label': label}
		
		if self.transform:
			sample = self.transform(sample)

		return sample
	
'''
Subclass of NematicsDataset to give sequences of data
'''
class NematicsSequenceDataset(NematicsDataset):
	def __init__(self, 
				 root_dir, 
				 frames_per_seq,
				 transform=None,
				 label_info=get_folder_info,
				 validation_split=0.2,
				 force_load=False):
		self.frames_per_seq = frames_per_seq
		super(NematicsSequenceDataset, self).__init__(
			root_dir,
			transform=transform,
			label_info=label_info,
			validation_split=validation_split,
			force_load=force_load)
	
	def split_indices(self, validation_split):
		self.load_sequences()
		print('Found %d sequences in %d folders' % (len(self), self.num_folders))
		
		split	= int(np.floor(validation_split * len(self)))
		indices = np.arange(len(self))
		np.random.shuffle(indices)
		self.train_indices, self.test_indices = indices[split:], indices[:split]

	def load_sequences(self):
		end_idx = np.array(self.dataframe.index) + self.frames_per_seq - 1
		fold_start = self.dataframe.folder
		fold_end = [-1] * len(fold_start)
		fold_end[:-(self.frames_per_seq-1)] = fold_start[end_idx[:-(self.frames_per_seq-1)]]
		invalid = fold_start != fold_end
		end_idx[invalid] = -2

		self.dataframe['end_idx'] = pd.Series(end_idx+1, index=self.dataframe.index)
		self.seq_idx = np.array(self.dataframe.index)[~invalid]

	def __len__(self):
		return len(self.seq_idx)
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		start_idx = self.seq_idx[idx]
		end_idx = self.dataframe.end_idx[start_idx]
		
		sample = self.get_image(start_idx)
		label = self.get_label(start_idx)

		sample = {	'x': np.zeros((self.frames_per_seq,) + sample.shape),
					'label': np.zeros((self.frames_per_seq,) + label.shape)}

		for i in range(start_idx, end_idx):
			sample['x'][i-start_idx] = self.get_image(i)
			sample['label'][i-start_idx] = self.get_label(i)

		if self.transform:
			sample = self.transform(sample)

		return sample

class Nematics3DDataset(NematicsDataset):
	def __init__(self, 
				 root_dir, 
				 transform=None,
				 label_info=get_folder_info,
				 validation_split=0.2,
				 force_load=False,
				 input_size=50):
		super(Nematics3DDataset, self).__init__(
			root_dir,
			transform=transform,
			label_info=label_info,
			validation_split=validation_split,
			force_load=force_load)
		self.input_size = input_size
	
	def get_image(self, idx):
		subdir = os.path.join(self.root_dir, self.dataframe.folder[idx])
		ind = self.dataframe.idx[idx]
		x = np.array([
			np.loadtxt(os.path.join(subdir, 'nx%d' % ind)),
			np.loadtxt(os.path.join(subdir, 'ny%d' % ind)),
			np.loadtxt(os.path.join(subdir, 'nz%d' % ind))])
		x = x.reshape([3, self.input_size, self.input_size, self.input_size])
		return x
