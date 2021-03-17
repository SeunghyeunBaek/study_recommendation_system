import torch
from torch import nn
from torch.autograd  import Variable
import torch.nn.init as weight_init

def weights_init(m):

    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

def MSEloss(inputs, targets, size_average=False):
  mask = targets != 0
  num_ratings = torch.sum(mask.float())
  criterion = nn.MSELoss(reduction='sum' if not size_average else 'mean')
  return criterion(inputs * mask.float(), targets), Variable(torch.Tensor([1.0])) if size_average else num_ratings