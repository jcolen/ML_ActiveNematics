import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from res_ae import ResidualFramePredictor

'''
Encoder and decoder use ConvNext architecture
'''

class LayerNorm2d(torch.nn.LayerNorm):
    r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if x.is_contiguous(memory_format=torch.contiguous_format):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), 
                self.normalized_shape, 
                self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x

class ConvNextBlock(nn.Module):
    def __init__(self, input_dim, output_dim):#, sd_prob=0.1):
        super().__init__()
        self.depth_conv = nn.Conv2d(input_dim, input_dim, 
                                    kernel_size=5, padding='same', groups=input_dim)
        self.norm = LayerNorm2d(input_dim)
        
        self.conv1 = nn.Conv2d(input_dim, 4*input_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(4*input_dim, output_dim, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(p=0.2)
            
    def forward(self, x):
        x = self.depth_conv(x)
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return x

class ConvNextEncoder(nn.Module):
    def __init__(self, 
                 input_dims,
                 stage_dims,
                 num_latent):
        super().__init__()
        self.downsample_blocks = nn.ModuleList()

        for i in range(len(stage_dims)-1):
            stage = nn.Sequential(
                LayerNorm2d(stage_dims[i][-1]),
                nn.Conv2d(stage_dims[i][-1], stage_dims[i+1][0], kernel_size=2, stride=2)
            )
            self.downsample_blocks.append(stage)

        stage = nn.Sequential(
            LayerNorm2d(stage_dims[-1][-1]),
            nn.Conv2d(stage_dims[-1][-1], num_latent, kernel_size=2, stride=2)
        )
        self.downsample_blocks.append(stage)

        self.stages = nn.ModuleList()
        for i in range(len(stage_dims)):
            stage = nn.Sequential(
                *[ConvNextBlock(stage_dims[i][j], stage_dims[i][j+1]) \
                  for j in range(len(stage_dims[i])-1)]
            )
            self.stages.append(stage)

        self.readin = nn.Conv2d(input_dims, stage_dims[0][0], kernel_size=1)
    
    def forward(self, x):
        x = self.readin(x)
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            x = self.downsample_blocks[i](x)
        return x
    
class ConvNextDecoder(nn.Module):
    def __init__(self,
                 output_dims,
                 stage_dims,
                 latent_dims):
        super().__init__()
        self.upsample_blocks = nn.ModuleList()

        stage = nn.Sequential(
            LayerNorm2d(latent_dims),
            nn.ConvTranspose2d(latent_dims, stage_dims[-1][-1], kernel_size=2, stride=2)
        )
        self.upsample_blocks.append(stage)

        for i in range(1, len(stage_dims)):
            stage = nn.Sequential(
                LayerNorm2d(stage_dims[-i][0]),
                nn.ConvTranspose2d(stage_dims[-i][0], stage_dims[-(i+1)][-1], kernel_size=2, stride=2)
            )
            self.upsample_blocks.append(stage)

        self.stages = nn.ModuleList()
        for i in reversed(range(len(stage_dims))):
            stage = nn.Sequential(
                *[ConvNextBlock(stage_dims[i][j], stage_dims[i][j-1]) \
                  for j in reversed(range(1, len(stage_dims[i])))]
            )
            self.stages.append(stage)

        self.readout = nn.Conv2d(stage_dims[0][0], output_dims, kernel_size=1)

    def forward(self, x):
        for i in range(len(self.stages)):
            x = self.upsample_blocks[i](x)
            x = self.stages[i](x)
        x = self.readout(x)
        return x

'''
Recurrent layers using ConvNext instead of Conv2d
The internal state is a latent vector defined at each point in space
For an input sequence of size [T, C, H0, W0]
    1. Encode a latent sequence [T, L, H1, W1]
    2. Process sequence into current state [1, L, H1, W1]
Given this current state, to evolve an input [C, H0, W0]
    1. Update local state LSTM -> [1, L, H1, W1]
    2. Update local state Conv -> [1, L, H1, W1]
'''
class ResidualConvNextNetwork(nn.Module):
    def __init__(self, vector_size=32, num_lstm_layers=2):
        super().__init__()
        self.vector_size = vector_size
        self.num_lstm_layers = num_lstm_layers
    
        self.lstm = nn.LSTM(
            input_size=vector_size,
            hidden_size=2*vector_size, #Inverse bottleneck approach
            proj_size=vector_size,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        self.conv = ConvNextBlock(vector_size, vector_size)


    def forward(self, x, hiddens=None):
        b, t, c, h, w = x.shape
        
        #Update local state using LSTM along time axis ONLY
        x = x.permute(0, 3, 4, 1, 2).reshape([-1, t, self.vector_size])
        rnn_out, hiddens = self.lstm(x, hiddens)
        x = x + rnn_out

        # Update local state using ConvNext along space axis
        x = x.reshape([b, h, w, t, self.vector_size])
        x = x.permute(0, 3, 4, 1, 2).reshape([b*t, self.vector_size, h, w])
        cnn_out = self.conv(x)
        x = x + cnn_out

        # Reshape to original dimensions
        x = x.reshape([b, t, self.vector_size, h, w])
        return x, hiddens

class ConvNextFramePredictor(ResidualFramePredictor):
    """ Key changes from original model
            - ConvNext encoder/decoder and a wider latent feature set
            - AdamW instead of Adam optimizer for weight decay
            - Train on extended sequence predictions (lookback 7, lookahead 3)
                - Explicitly separate burn-in from prediction to show how this is done
    """
    def __init__(self,
                 in_channels=2,
                 num_latent=32,
                 stage_dims=[[32,32], [64,64]], 
                 num_lstm_layers=2,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_latent = num_latent
        self.stage_dims = stage_dims
        self.num_lstm_layers = num_lstm_layers
        
        self.encoder = ConvNextEncoder(in_channels, stage_dims, num_latent)
        self.decoder = ConvNextDecoder(in_channels, stage_dims, num_latent)
        self.recurrent = ResidualConvNextNetwork(num_latent, num_lstm_layers)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)


