import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CNNCell(nn.Module):
    def __init__(self, 
                 input_channels, 
                 output_channels,
                 kernel_size=3):
        super().__init__()
        
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding='same')
        self.bn = nn.BatchNorm2d(output_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

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
                 input_size=32, # Spatial extent of input region
                 output_dims=1, # Number of parameters to estimate
                 conv_size=32,  # Number of CNN features to use
                 num_convs=1,   # Number of CNN layers
                 kernel_size=3, # Convolutional kernel_size
                 rnn_size=32,   # Number of RNN features to use
                 num_rnns=1,    # Number of RNN layers
                 fcnn_size=32,  # Number of dense features to use
                 num_fcnn=1,    # Number of dense layers
                 dropout=0.1,   # Dropout rate
                 ):
        super().__init__()
        self.input_size = input_size
        
        # Convolutional network module
        self.cnn_cells = nn.ModuleList()
        self.cnn_cells.append(CNNCell(1, conv_size, kernel_size=kernel_size))
        for i in range(num_convs-1):
            self.cnn_cells.append(CNNCell(conv_size, conv_size, kernel_size=kernel_size))

        # Dropout after CNN features
        self.dropout = nn.Dropout(p=dropout)

        # Get recurrent vector size
        dummy = torch.ones([1, 1, input_size, input_size])
        for cell in self.cnn_cells:
            dummy = cell(dummy)
        vector_size = torch.numel(dummy)

        logger.info(f'First LSTM layer sees a vector of size {vector_size}')

        # Recurrent network module
        self.rnn_cells = nn.ModuleList()
        self.rnn_cells.append(nn.LSTM(vector_size, rnn_size, num_layers=1, batch_first=True)) # project onto low-dim lstm space
        for i in range(num_rnns - 1):
            self.rnn_cells.append(nn.LSTM(rnn_size, rnn_size, num_layers=1, batch_first=True))

        # Fully connected network module
        self.fcnn_cells = nn.ModuleList()
        self.fcnn_cells.append(nn.Linear(rnn_size, fcnn_size)) # project onto fcnn feature space
        for i in range(num_fcnn - 1):
            self.fcnn_cells.append(nn.Linear(fcnn_size, fcnn_size))
        self.fcnn_cells.append(nn.Linear(fcnn_size, output_dims))

        # Apply weight initialization all at once
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)
    
    def forward(self, x):
        # Apply CNN layers to all time points
        b, t, c, h0, w0 = x.size()
        x = x.reshape([b*t, c, h0, w0])
        for cell in self.cnn_cells:
            x = cell(x) # CNNCell contains its own activation function

        # Apply dropout
        x = self.dropout(x)
        
        # Apply RNN layers to the time sequence
        x = x.reshape([b, t, -1])
        for cell in self.rnn_cells:
            x, _ = cell(x)

        x = x[:, -1] # Only keep final time point

        # Apply fully connected layers
        for cell in self.fcnn_cells:
            x = cell(x)
            x = F.tanh(x) # Apply nonlinear activation as well

        return x
    
    def batch_step(self, batch, criterion, optimizer, device):
        """ Deprecated - this is now done in the trainer script
        """
        if self.training:	optimizer.zero_grad()
        x, label = batch['x'].to(device), batch['label'].to(device)
        y = self(x)
        loss = criterion(y, label)
        if self.training:
            loss.backward()
            optimizer.step()
        return loss.item()
    
    def batch_predict(self, batch, device, n=10):
        """ Deprecated - this is now done in the trainer script
        """
        x, label = batch['x'].to(device), batch['label'].to(device)
        y = self(x)
        return label, y
        
