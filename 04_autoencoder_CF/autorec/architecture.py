
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.init as weight_init


class SimpleAutoEncoder(nn.Module):
  def __init__(self, num_inputs, num_hiddens, kind='sigmoid', dropout=None):
    super(SimpleAutoEncoder, self).__init__()
    # encoder -> hidden -> decoder
    # input -> hidden -> output
    # input -> hidden : encoder
    # hidden -> output = input : decoder
    self.encoder = nn.Sequential(nn.Linear(num_inputs, num_hiddens), self.activation(kind))
    self.decoder = nn.Sequential(nn.Linear(num_hiddens, num_inputs), self.activation(kind))  

  def activation(self, kind):
    if kind == 'selu':
      return nn.SELU()
    elif kind == 'relu':
      return nn.ReLU()
    elif kind == 'relu6':
      return nn.ReLU6()
    elif kind == 'sigmoid':
      return nn.Sigmoid()
    elif kind == 'tanh':
      return nn.Tanh()
    elif kind == 'elu':
      return nn.ELU()
    elif kind == 'lrelu':
      return nn.LeakyReLU()
    elif kind == 'none':
      return input
    else:
      raise ValueError('Unknown non-linearity type')

  def forward(self, x):
    return self.decoder(self.encoder(x))



class DeepAutoEncoder(nn.Module):
  def __init__(self, num_hiddens, num_layers, dropout=None, nn_type='diamond'):
    super(DeepAutoEncoder, self).__init__()
    # input -> hidden -> output
    # input -> hidden(10) -> ... -> hidden(10) -> output = input
    self.encoder, self.decoder = self.generate_layers(num_hiddens, num_layers, dropout, nn_type)
  
  def forward(self, x):
    return self.decoder(self.encoder(x))
  
  def generate_layers(self, num_hiddens, num_layers, dropout=None, nn_type='diamond'):
    # hidden layers -> [50, 25, 12, 6, 12, 25, 50], [100 50 100] -> 100, 50, 60, 50 100 
    if nn_type == 'diamond':
      encoder_modules = []
      decoder_modules = []

      hidden_layers = []
      temp = num_hiddens
      for idx, x in enumerate(range(num_layers)):
        if idx == 0:
          hidden_layers.append(temp)
        else:
          hidden_layers.append(int(temp/2))
        temp = temp/2
      hidden_layers = [x for x in hidden_layers if x > 10]
      
      # encoder
      for idx, num_hiddens in enumerate(hidden_layers):
        if idx < len(hidden_layers)-1:
          encoder_modules.append(nn.Linear(hidden_layers[idx], hidden_layers[idx+1], bias=True))
          encoder_modules.append(nn.Sigmoid())

      # decoder
      hidden_layers = list(reversed(hidden_layers))
      for idx, num_hidden in enumerate(hidden_layers):
        if idx < len(hidden_layers)-1:
          decoder_modules.append(nn.Linear(hidden_layers[idx], hidden_layers[idx+1], bias=True))
          decoder_modules.append(nn.Identity())

    # num_hidden = 50, num_layers = 3 ->  input_dim -> [50, 50, 50] -> output_dim = input_dim 
    elif nn_type == 'constant':
      hidden_layers = [num_hiddens] * num_layers
      for idx, enc in enumerate(hidden_layers):
        if idx < num_layers-1:
          encoder_modules.append(nn.Linear(hidden_layers[idx], hidden_layers[idx+1], bias=True))
          encoder_modules.append(nn.Sigmoid())
          decoder_modules.append(nn.Linear(hidden_layers[idx], hidden_layers[idx+1], bias=True))
          decoder_modules.append(nn.Identity())

    if dropout is not None:    
      encoder_modules = [x for y in (encoder_modules[i:i+2] + [nn.Dropout(dropout)] * (i < len(encoder_modules) - 1) 
                          for i in range(0, len(encoder_modules), 2)) for x in y]
      decoder_modules = [x for y in (decoder_modules[i:i+2] + [nn.Dropout(dropout)] * (i < len(decoder_modules) - 1)
                          for i in range(0, len(decoder_modules), 2)) for x in y]

    encoder = nn.Sequential(*encoder_modules)
    decoder = nn.Sequential(*decoder_modules)
    
    return encoder, decoder