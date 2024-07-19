import torch.nn as nn
import torch.nn.functional as F

def activation(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'sigmoid':
        return torch.sigmoid
    elif activation == 'tanh':
        return torch.tanh
    elif activation == 'softmax':
        return lambda x: F.softmax(x, dim=1)
    elif activation == 'leaky_relu':
        return F.leaky_relu
    elif activation == 'linear':
        return lambda x: x
    else:
        raise NotImplementedError(f"Activation {activation} not implemented")
