import numpy as np
import torch
import torch.nn.functional as F
import os
import argparse
from time import time
from torchvision import transforms

import data_processing as dp
from simulation_datasets import NematicsSequenceDataset
from models import ParameterEstimator

class WeightedMSELoss(torch.nn.MSELoss):
	def __init__(self, size_average=None, reduce=None, reduction='mean', weights=None):
		super(WeightedMSELoss, self).__init__(size_average, reduce, reduction)
		self.weights = weights

	def forward(self, input, target):
		if self.weights is None:
			return F.mse_loss(input, target, reduction=self.reduction)
		else:
			return F.mse_loss(input * self.weights,
							  target * self.weights,
							  reduction=self.reduction)
	
def iterate_loader(model, loader, optimizer, criterion, device):
	loss = 0
	for i, batch in enumerate(loader):
		loss += model.batch_step(batch, criterion, optimizer, device=device)
		if args.trial and i == 1:	break
	return loss / i

def predict(model, loader, device, outfile, n=10):
	model.eval()
	with open(outfile, 'w') as fout:
		with torch.no_grad():
			for cnt, batch in enumerate(loader):
				labels, preds = model.batch_predict(batch, device, n=n)
				for i in range(labels.shape[0]):
					for j in range(labels.shape[1]):
						fout.write('%g\t' % labels[i, j])
					for j in range(preds.shape[1]):
						fout.write('%g\t' % preds[i, j])
					fout.write('\n')
				if args.trial and cnt == 1:	break

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--directory', type=str, 
		default='/home/jcolen/data/short_time_multi_parameter/') 
	parser.add_argument('-f', '--num_frames', type=int, default=8, 
		help='Number of frames used in RNN model predictions')
	parser.add_argument('--conv_channels', type=int, nargs='+', default=[1,32],
		help='Input/output channels for convolutional layers')
	parser.add_argument('--rnn_sizes', type=int, nargs='+', default=[32],
		help='Sizes of recurrent (LSTM) layers')
	parser.add_argument('--dense_sizes', type=int, nargs='+', default=[32],
		help='Sizes of dense layers')
	parser.add_argument('--kernel_size', type=int, default=3,
		help='Convolutional kernel size')
	parser.add_argument('--dropout', type=float, default=0.1,
		help='Dropout layer probability')
	parser.add_argument('--random_crop', type=int, default=32,
		help='Crop size of input image/volume')
	parser.add_argument('-e', '--epochs', type=int, default=100)
	parser.add_argument('-b', '--batch_size', type=int, default=32)
	parser.add_argument('--num_workers', type=int, default=6,
		help='Workers for file loading')
	parser.add_argument('--validation_split', type=float, default=0.2)
	parser.add_argument('--ndims', type=int, default=2)
	parser.add_argument('--force_new', action='store_true',
		help='Force overwrite of model')
	parser.add_argument('--trial', action='store_true',
		help='Testing utility. Stop epoch after 2 batches')
	args = parser.parse_args()

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	pin_memory = True
	print(device)

	if not os.path.exists('models'):
		os.mkdir('models')
	if not os.path.exists('predictions'):
		os.mkdir('predictions')

	if args.ndims == 2:
		transform = transforms.Compose([
			dp.Sin2t(),
			dp.RandomFlip(),
			dp.RandomShift(),
			dp.RandomRotation(45.),
			dp.RandomCrop(args.random_crop),
			dp.AverageTimeLabel(),
			dp.ToTensor()])
		dataset = NematicsSequenceDataset(args.directory, args.num_frames, transform=transform, 
			validation_split=args.validation_split)
	else:
		transform = transforms.Compose([
			dp.Qij(),
			dp.RandomFlip(),
			dp.RandomShift(),
			dp.RandomRotation(45.),
			dp.RandomCrop(args.random_crop),
			dp.ToTensor()])
		dataset = Nematics3DDataset(args.directory, transform=transform, validation_split=args.validation_split)

	train_loader = dataset.get_loader(dataset.train_indices, args.batch_size, args.num_workers, pin_memory)
	test_loader = dataset.get_loader(dataset.test_indices, args.batch_size, args.num_workers, pin_memory)
	
	model = ParameterEstimator(
		input_size=args.random_crop,
		conv_channels=args.conv_channels,
		rnn_sizes=args.rnn_sizes if args.ndims == 2 else None,
		dense_sizes=args.dense_sizes,
		params=dataset.label_names,
		ndims=args.ndims,
	)

	if not args.force_new and os.path.exists('models/%s' % model.name):
		print('Loading from file')
		model_info = torch.load('models/%s' % model.name)
		model.load_state_dict(model_info['state_dict'])
		loss_min = model_info['loss']
		losses = model_info['losses']
	else:
		loss_min = np.Inf
		losses = []
	model.to(device)
	best_epoch = len(losses)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.92)
	weights = dataset.get_average_labels()
	weights = np.max(weights) / weights
	criterion = WeightedMSELoss(weights=torch.tensor(weights).to(device))
	print(model.name)

	#Training
	patient = 20
	for epoch in range(args.epochs):
		if epoch - best_epoch >= patient:
			print('early stop at epoch %g' % best_epoch)
			break
		
		t_ini = time()
		loss_train = iterate_loader(model.train(), train_loader, optimizer, criterion, device)
		loss_test	= iterate_loader(model.eval(), test_loader, optimizer, criterion, device)
		t_end = time()
		
		print('Epoch %g: train: %g, test: %g\ttime=%g' % \
			(epoch, loss_train, loss_test, t_end - t_ini), flush=True)
		scheduler.step()
		losses.append(loss_test)
		if loss_test < loss_min:
			torch.save(
				{'state_dict': model.state_dict(),
				 'loss': loss_test,
				 'losses': losses},
				 'models/%s' % model.name)
			predict(model, test_loader, device, 'predictions/%s' % model.name)
			loss_min = loss_test
			best_epoch = epoch
