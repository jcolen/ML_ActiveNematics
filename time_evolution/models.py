import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Encoder built of strided convolutional layers
'''

class CnnCell(nn.Module):
	def __init__(self, in_channel, out_channel):
		super(CnnCell, self).__init__()
		
		self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
		self.bn   = nn.BatchNorm2d(out_channel)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = F.tanh(x)
		return x

class Encoder(nn.Module):
	def __init__(self, channels):
		super(Encoder, self).__init__()

		self.cells = nn.ModuleList()
		for i in range(len(channels)-1):
			self.cells.append(CnnCell(channels[i], channels[i+1]))
		
	def forward(self, x):
		for i in range(len(self.cells)):
			x = self.cells[i](x)
		return x
		
'''
Decoder built of transposed convolutional layers
'''

class DeCnnCell(nn.Module):
	def __init__(self, in_channel, out_channel):
		super(DeCnnCell, self).__init__()
		
		self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
		self.bn		= nn.BatchNorm2d(out_channel)

	def forward(self, x):
		x = self.deconv(x)
		x = self.bn(x)
		x = F.tanh(x)
		return x

class Decoder(nn.Module):
	def __init__(self, channels):
		super(Decoder, self).__init__()

		self.cells = nn.ModuleList()
		for i in range(1, len(channels)):
			self.cells.append(DeCnnCell(channels[-i], channels[-(i+1)]))

	def forward(self, x):
		for i in range(len(self.cells)):
			x = self.cells[i](x)
		return x

'''
Recurrent layers compute dynamics at the autoencoder bottleneck
Recurrent layers are built using LSTM or GRU blocks
'''

class GRUCell(nn.Module):
	def __init__(self, vector_size, num_layers):
		super(GRUCell, self).__init__()
		self.gru = nn.GRU(
			vector_size,
			vector_size,
			num_layers=num_layers,
			batch_first=True
		)

	def forward(self, x):
		x, _ = self.gru(x)
		return x

class LSTMCell(nn.Module):
	def __init__(self, vector_size, num_layers):
		super(LSTMCell, self).__init__()
		self.lstm = nn.LSTM(
			vector_size,
			vector_size,
			num_layers=num_layers,
			batch_first=True
		)

	def forward(self, x):
		x, _ = self.lstm(x)
		return x

class RecurrentNetwork(nn.Module):
	def __init__(self, num_cells, vector_size, layers_per_cell, cell='lstm'):
		super(RecurrentNetwork, self).__init__()
		self.name = 'recurrent_%dlstm_l%d' % (num_cells, layers_per_cell)
		self.cells = nn.ModuleList()

		cell_dict = {
			'lstm': LSTMCell,
			'gru': GRUCell
		}

		for i in range(num_cells):
			self.cells.append(cell_dict[cell](vector_size, layers_per_cell))

	def forward(self, x):
		for i in range(len(self.cells)):
			x = self.cells[i](x)
		return x

class ResidualNetwork(RecurrentNetwork):
	def __init__(self, num_cells, vector_size, layers_per_cell, cell='lstm'):
		super(ResidualNetwork, self).__init__(num_cells, vector_size, layers_per_cell, cell)
		self.name = 'resnet_%dlstm_l%d' % (num_cells, layers_per_cell)
	
	def forward(self, x):
		rnn_out = self.cells[0](x)
		for i in range(1, len(self.cells)):
			rnn_out = self.cells[i](rnn_out)
		x = x + rnn_out
		return x

class LinearResidualNetwork(RecurrentNetwork):
	def __init__(self, num_cells, vector_size, layers_per_cell, cell='gru'):
		super(LinearResidualNetwork, self).__init__(num_cells, vector_size, layers_per_cell, cell)
		self.name = 'linres_%dgru_l%d' % (num_cells, layers_per_cell)

		self.linear1 = nn.Linear(vector_size, vector_size)
		self.linear2 = nn.Linear(vector_size, vector_size)

	def forward(self, x):
		rnn_out = self.cells[0](x)
		for i in range(1, len(self.cells)):
			rnn_out = self.cells[i](rnn_out)
		x = x * F.sigmoid(self.linear1(rnn_out)) + F.tanh(self.linear2(rnn_out))
		return x

'''
Frame predictor consists of an encoder, deocder, and recurrent layer
'''

class FramePredictor(nn.Module):
	def __init__(self, 
				 autoencoder_channels,
				 recurrent='residual',
				 input_size=48,
				 num_lstms=2,
				 layers_per_lstm=1):
		super(FramePredictor, self).__init__()
		self.encoder = Encoder(autoencoder_channels)
		self.decoder = Decoder(autoencoder_channels)

		rec_dict = {
			'recurrent':	RecurrentNetwork,
			'residual': 	ResidualNetwork,
			'linear_res':	LinearResidualNetwork,
		}

		vector_size = int(autoencoder_channels[-1] * (input_size / 2**(len(autoencoder_channels)-1))**2)
		self.recurrent = rec_dict[recurrent](num_lstms, vector_size, layers_per_lstm)

		self.input_size = input_size
		self.name = 'frame_predictor_input%d_c%s_%s.pt' % (
			self.input_size,
			','.join([str(c) for c in autoencoder_channels]),
			self.recurrent.name)

	def forward(self, x):
		b, t, c, h, w = x.size()
		x = x.contiguous().view([b * t, c, h, w])
		x = self.encoder(x)
		_, fc, fh, fw = x.size()
		
		x = x.view([b, t, -1])
		x = self.recurrent(x)
		
		x = x.contiguous().view([b * t, fc, fh, fw])
		x = self.decoder(x)
		x = x.view([b, t, c, h, w])[:, -1]

		#Normalize output
		norm = x.norm(p=2, dim=1, keepdim=True)
		x = x.div(norm.expand_as(x))

		return x
	
	def batch_step(self, batch, criterion, optimizer, device):
		if self.training:	optimizer.zero_grad()
		frames = batch['image'].to(device)
		nextframe = frames[:, -1]
		sequences = frames[:, :-1]
		preds = self(sequences)
		loss = criterion(preds, nextframe)
		if self.training:
			loss.backward()
			optimizer.step()
		return loss.item()
	
	def freeze_recurrent(self):
		for param in self.recurrent:
			param.requires_grad=False
	
	def named_grad_parameters(self):
		params = []
		for name, param in self.named_parameters():
			if param.requires_grad:
				params.append(param)
		return params
