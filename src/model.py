import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class AdversarialNeuralCryptoNet(nn.Module):
    '''
    Standard Convolutional layers setup used by Alice, Bob and Eve.
    Input: 4d tensor, (batch_size,1,msg_len+key_len,1)
    Output: 4d tensor, (batch_size,1,msg_len,1)
    '''
    def __init__(self):
        super(AdversarialNeuralCryptoNet,self).__init__()

    def forward(self,x):
        pass
