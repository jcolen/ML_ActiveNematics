import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNCell(nn.Module):
	def __init__(self, 
				 input_channels, 
				 output_channels,
				 kernel_size=3,
				 ndims=2):
		super(CNNCell, self).__init__()

		conv_dict = {
			2: nn.Conv2d,
			3: nn.Conv3d,
		}

		bn_dict = {
			2: nn.BatchNorm2d,
			3: nn.BatchNorm3d,
		}
		
		pool_dict = {
			2: nn.MaxPool2d,
			3: nn.MaxPool3d,
		}
		

		padding = int((-1. / 2 + kernel_size) // 2)	
		self.conv = conv_dict[ndims](input_channels, output_channels, kernel_size=kernel_size, padding=padding)
		self.bn = bn_dict[ndims](output_channels)
		self.pool = pool_dict[ndims](kernel_size=2)

		nn.init.xavier_uniform_(self.conv.weight)
		nn.init.constant_(self.conv.bias, 0.)
	
	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = F.tanh(x)
		x = self.pool(x)
		return x

class ParameterEstimator(nn.Module):
	def __init__(self,
				 input_size=32,
				 conv_channels=[1,32],
				 kernel_size=3,
				 rnn_sizes=None,
				 dense_sizes=[32],
				 params=['z'],
				 dropout=0.1,
				 ndims=2):
		super(ParameterEstimator, self).__init__()
		self.input_size = input_size
		self.ndims = ndims
		
		self.name = 'parameter_estimator_%s_%dD_input%d_k3c%s_lstm%s_fc%s.pt' % (
			str(params),
			self.ndims,
			self.input_size,
			','.join([str(c) for c in conv_channels]),
			','.join([str(r) for r in rnn_sizes]),
			','.join([str(d) for d in dense_sizes]))
		
		self.cnn_cells = nn.ModuleList()
		self.rnn_cells = nn.ModuleList()
		self.dense_cells = nn.ModuleList()

		self.flattened_nfeatures = self.input_size

		for i in range(len(conv_channels)-1):
			self.cnn_cells.append(CNNCell(
				conv_channels[i], conv_channels[i+1],
				kernel_size=kernel_size, ndims=ndims))
			self.flattened_nfeatures = self.flattened_nfeatures // 2

		self.flattened_nfeatures = int(conv_channels[-1] * (self.flattened_nfeatures**ndims))
		self.dropout = nn.Dropout(p=dropout)

		if rnn_sizes is not None:
			rnn_sizes = [self.flattened_nfeatures,] + rnn_sizes
			for i in range(len(rnn_sizes) - 1):
				self.rnn_cells.append(nn.LSTM(rnn_sizes[i], rnn_sizes[i+1], num_layers=1, batch_first=True))
				for name, param in self.rnn_cells[-1].named_parameters():
					if 'bias' in name:
						nn.init.constant_(param, 0.)
					elif 'weight' in name:
						nn.init.xavier_uniform_(param)
			dense_sizes = [rnn_sizes[-1],] + dense_sizes
		dense_sizes = dense_sizes + [len(params),]

		for i in range(len(dense_sizes) - 1):
			self.dense_cells.append(nn.Linear(dense_sizes[i], dense_sizes[i+1]))
			nn.init.xavier_uniform_(self.dense_cells[-1].weight)
			nn.init.constant_(self.dense_cells[-1].bias, 0.)

	def forward(self, x):
		if len(self.rnn_cells) > 0:
			 b, t, c = x.size()[:-self.ndims]
			 dims = x.size()[-self.ndims:]
			 x = x.view(b*t, c, *dims)

		for cell in self.cnn_cells:
			x = cell(x)

		if len(self.rnn_cells) > 0:
			x = x.view(b, t, -1)
			x = self.dropout(x)
			for cell in self.rnn_cells:
				x, _ = cell(x)
			x = x[:, -1]
		else:
			x = x.view(-1, self.flattened_nfeatures)
			x = self.dropout(x)

		for cell in self.dense_cells:
			x = cell(x)
			x = F.tanh(x)

		return x
	
	def batch_step(self, batch, criterion, optimizer, device):
		if self.training:	optimizer.zero_grad()
		x, label = batch['x'].to(device), batch['label'].to(device)
		y = self(x)
		loss = criterion(y, label)
		if self.training:
			loss.backward()
			optimizer.step()
		return loss.item()
	
	def batch_predict(self, batch, device, n=10):
		x, label = batch['x'].to(device), batch['label'].to(device)
		y = self(x)
		return label, y
		
