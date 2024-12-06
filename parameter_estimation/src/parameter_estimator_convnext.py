import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
Encoder uses ConvNext architecture
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
    def __init__(self, input_dim, output_dim):
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

'''
Recurrent layers using ConvNext instead of Conv2d
The internal state is a latent vector defined at each point in space
For an input sequence of size [T, C, H0, W0]
    1. Encode a latent sequence [T, L, H1, W1]
    2. Process sequence into current state [1, L, H1, W1]
Given this current state, to estimate an input [C, H0, W0]
    1. Update local state LSTM -> [1, L, H1, W1]
    2. Update local state Conv -> [1, L, H1, W1]
'''
class RecurrentConvNextLayer(nn.Module):
    def __init__(self, vector_size=32, num_layers=1):
        super().__init__()
        self.vector_size = vector_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=vector_size,
            hidden_size=2*vector_size, #Inverse bottleneck approach
            proj_size=vector_size,
            num_layers=num_layers,
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

'''
Pseudo-fcnn layer - Note that the RNN now applies equivalently to each spatiotemporal point. Rather than apply a fcnn layer to this flattened vector, we will apply a dense layer to each spatiotemporal point using Conv layers with kernel size 1. Finally, we'll average to estimate parameters for the region
'''
class PseudoFCNNLayer(nn.Module):
    def __init__(self, input_size=32, fcnn_size=32, output_dims=1, num_fcnn=1):
        super().__init__()

        self.fcnn_size = fcnn_size
        self.output_dims = output_dims
        self.num_fcnn = num_fcnn

        self.fcnn_cells = nn.ModuleList()
        self.fcnn_cells.append(nn.Conv3d(input_size, fcnn_size, kernel_size=1))
        for i in range(num_fcnn - 1):
            self.fcnn_cells.append(nn.Conv3d(fcnn_size, fcnn_size, kernel_size=1))
        self.fcnn_cells.append(nn.Conv3d(fcnn_size, output_dims, kernel_size=1))

    def forward(self, x, pool=True):
        b, c, t, h, w = x.shape
        for cell in self.fcnn_cells[:-1]:
            x = cell(x)
            x = F.tanh(x) # Apply nonlinear activation
        x = self.fcnn_cells[-1](x) # Apply last layer

        return x

class ConvNextParameterEstimator(nn.Module):
    """ Key changes from original model
            - ConvNext encoder
            - Smaller LSTM module with spatial independence
            - Final layer uses global average pooling instead of a large linear layer
            - AdamW instead of Adam optimizer for weight decay
    """
    def __init__(self,
                 input_size=32, # Spatial extent of input region
                 output_dims=2, # Number of parameters to estimate
                 conv_size=32,  # Number of CNN features to use
                 num_convs=1,   # Number of CNN layers
                 kernel_size=5, # Convolutional kernel_size
                 rnn_size=32,   # Number of RNN features to use
                 num_rnns=1,    # Number of RNN layers
                 fcnn_size=32,  # Number of dense features to use
                 num_fcnn=1,    # Number of dense layers
                 dropout=0.1,   # Dropout rate
                 ):
        super().__init__()

        # Convolutional network module            
        self.cnn_cells = ConvNextEncoder(
            input_dims=1, 
            stage_dims=[[conv_size, conv_size], [conv_size, conv_size]],
            num_latent=rnn_size
        )

        # Recurrent network module
        self.rnn_cells = RecurrentConvNextLayer(
            vector_size=rnn_size,
            num_layers=num_rnns
        )

        # Fully connected network module
        self.fcnn_cells = PseudoFCNNLayer(
            input_size=rnn_size,
            fcnn_size=fcnn_size,
            output_dims=output_dims,
            num_fcnn=num_fcnn
        )

        # Apply weight initialization all at once
        self.apply(self._init_weights)

    def forward(self, x, pool=True):
        # Apply CNN layers
        b, t, c0, h0, w0 = x.size()
        x = x.reshape([b*t, c0, h0, w0])
        x = self.cnn_cells(x) #ConvNext cells contain dropout
        _, c1, h1, w1 = x.shape
        
        # Apply LSTM layers
        x = x.reshape([b, t, c1, h1, w1])
        x, _ = self.rnn_cells(x)

        # Apply fcnn layers
        x = x.permute(0, 2, 1, 3, 4) # [B, C, T, H, W]
        x = self.fcnn_cells(x)

        if pool:
            return torch.mean(x, dim=(-3, -2, -1))
        else:
            return x

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
    
        
        
        