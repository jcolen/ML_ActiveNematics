import argparse
import os
import numpy as np
from time import time

import torch
from torchvision import transforms

from simulation_datasets import NematicsSequenceDataset
from models import FramePredictor
import data_processing as dp

criteria = {
	'mse': torch.nn.MSELoss,
	'l1': torch.nn.L1Loss
}

def iterate_loader(model, loader, optimizer, criterion, device):
	loss = 0
	for i, batch in enumerate(loader):
		loss += model.batch_step(batch, criterion, optimizer, device)
		if args.trial and i == 1:	break
	return loss / i

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--directory', type=str, default='/home/jcolen/data/deltat_10s')
	parser.add_argument('-e', '--epochs', type=int, default=100)
	parser.add_argument('-b', '--batch_size', type=int, default=64)
	parser.add_argument('-c', '--criterion', choices=criteria.keys(), default='l1')
	parser.add_argument('--channels', type=int, nargs='+', default=[2, 4, 6])
	parser.add_argument('--num_lstms', type=int, default=2)
	parser.add_argument('--num_frames', type=int, default=8)
	parser.add_argument('--crop_size', type=int, default=48)
	parser.add_argument('--num_workers', type=int, default=2)
	parser.add_argument('--freeze_recurrent', action='store_true')
	parser.add_argument('--validation_split', type=float, default=0.2)
	parser.add_argument('--force_new', action='store_true',
		help='Force overwrite of model')
	parser.add_argument('--trial', action='store_true',
		help='Testing utility. Stop epoch after 2 batches')
	args = parser.parse_args()

	#GPU or CPU
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	pin_memory = True
	print(device)

	if not os.path.exists('models'):
		os.mkdir('models')
	
	# Dataset
	transform = transforms.Compose([
		dp.SinCos(),
		dp.RandomFlip(0.5),
		dp.RandomTranspose(0.5),
		dp.RandomShift(0.5),
		dp.RandomCrop(args.crop_size),
		dp.ToTensor()])
	dataset = NematicsSequenceDataset(args.directory, args.num_frames, transform=transform, 
		validation_split=args.validation_split)
	train_loader = dataset.get_loader(dataset.train_indices, args.batch_size, args.num_workers, pin_memory)
	test_loader = dataset.get_loader(dataset.test_indices, args.batch_size, args.num_workers, pin_memory)

	# Models
	model = FramePredictor(args.channels,
						   recurrent='residual',
						   #recurrent='linear_res',
						   input_size=args.crop_size,
						   num_lstms=args.num_lstms)
	if not args.force_new and os.path.exists('models/%s' % model.name):
		print('Loaded from file')
		model_info = torch.load('models/%s' % model.name)
		model.load_state_dict(model_info['state_dict'])
		loss_min = model_info['loss']
		losses = model_info['losses']
	else:
		loss_min = np.Inf
		losses = []
	if args.freeze_recurrent:	model.freeze_recurrent()
	model.to(device)
	best_epoch = len(losses)
	optimizer = torch.optim.Adam(model.named_grad_parameters(), lr=0.001)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.92)
	criterion = criteria[args.criterion]()
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
			best_epoch = epoch
			loss_min = loss_test
