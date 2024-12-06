import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Encoder built of strided convolutional layers
'''

class CnnCell(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
        self.bn   = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.tanh(x)
        return x

class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()

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
        super().__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
        self.bn		= nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = F.tanh(x)
        return x

class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.cells = nn.ModuleList()
        for i in range(1, len(channels)):
            self.cells.append(DeCnnCell(channels[-i], channels[-(i+1)]))

    def forward(self, x):
        for i in range(len(self.cells)):
            x = self.cells[i](x)
        return x

'''
Recurrent layers compute dynamics at the autoencoder bottleneck
'''
class ResidualNetwork(nn.Module):
    def __init__(self, vector_size, num_cells=2, num_layers=1):
        super().__init__()
        self.num_cells = num_cells
        self.vector_size = vector_size
        self.num_layers = num_layers
    
        self.cells = nn.ModuleList()
        for i in range(num_cells):
            self.cells.append(nn.LSTM(
                input_size=vector_size,
                hidden_size=vector_size,
                num_layers=num_layers,
                batch_first=True
            ))

    def forward(self, x, hiddens=None):
        '''
        Added for forward compatibility with ConvNext models
            - hiddens allows tracking and passing hidden states as arguments to the recurrent cells
            - reshaping is moved inside this module rather than done by FramePredictor
            - removed the silly LSTMCell which is just a wrapper around torch.nn.LSTM? 
                No idea what the purpose of that was. There is such a thing as *too* modular.
        '''
        input_shape = x.shape #B, T, ...
        x = x.reshape([*input_shape[:2], self.vector_size])
        
        if hiddens is None:
            hiddens = [None for i in range(self.num_cells)]
        for i, cell in enumerate(self.cells):
            rnn_out, (hn, cn) = cell(x, hiddens[i])
            x = x + rnn_out
            hiddens[i] = (hn, cn)

        x = x.reshape(input_shape)
        return x, hiddens

'''
Frame predictor consists of an encoder, decoder, and recurrent layer
'''

class ResidualFramePredictor(nn.Module):
    def __init__(self,
                 autoencoder_channels=[2,4,6],
                 recurrent_class=ResidualNetwork,
                 input_size=48,
                 num_lstm_cells=2,
                 num_lstm_layers=1):
        super(ResidualFramePredictor, self).__init__()
        self.encoder = Encoder(autoencoder_channels)
        self.decoder = Decoder(autoencoder_channels)

        if recurrent_class == ResidualNetwork:
            dummy = torch.ones([1, autoencoder_channels[0], input_size, input_size])
            with torch.no_grad():
                vector_size = torch.numel(self.encoder(dummy))
        else: # Forward compatibility with ConvNext models
            vector_size = autoencoder_channels[-1]

        self.recurrent = recurrent_class(vector_size, num_lstm_cells, num_lstm_layers)

        self.input_size = input_size
        self.num_latent = autoencoder_channels[-1]
        self.num_lstm_cells = num_lstm_cells
        self.num_lstm_layers = num_lstm_layers

    def forward(self, x, tmax=1):
        # Apply encoder to all time points
        b, t, c, h0, w0 = x.size()
        x = x.reshape([b * t, c, h0, w0])
        x = self.encoder(x)
        _, _, h1, w1 = x.shape

        x = x.reshape([b, t, self.num_latent, h1, w1])
        inputs = x[:, :-1]
        params = x[:, -1:]

        # Build internal state from initial sequence
        output, hiddens = self.recurrent(inputs)

        # Evolve using LSTM state after burn-in
        x_out = torch.zeros([b, tmax, self.num_latent, h1, w1], dtype=x.dtype, device=x.device)
        for i in range(tmax):
            params, hiddens = self.recurrent(params, hiddens)
            x_out[:, i:i+1] += params

        # Apply decoder to all time points
        x_out = x_out.reshape([b*tmax, self.num_latent, h1, w1])
        x_out = self.decoder(x_out)
        x_out = x_out.reshape([b, tmax, c, h0, w0])

        #Normalize output since sin^2 + cos^2 = 1
        norm = x_out.norm(p=2, dim=-3, keepdim=True)
        x_out = x_out.div(norm.expand_as(x_out))

        return x_out
    
    def batch_step(self, batch, criterion, optimizer, device):
        """ Deprecated - this is now done in the trainer script
        """
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
        for param in self.recurrent.parameters():
            param.requires_grad=False

    def freeze_spatial(self):
        for param in self.encoder.parameters():
            param.requires_grad=False
        for param in self.decoder.parameters():
            param.requires_grad=False
    
    def named_grad_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params
