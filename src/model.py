import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class CryptoNet_Alice(nn.Module):
    '''
    Standard Convolutional layers setup used by Alice.
    Input: 4d tensor, (batch_size,1,msg_len+key_len,1)
    Output: 4d tensor, (batch_size,1,msg_len,1)
    '''
    def __init__(self,n_input,n_output):
        super(CryptoNet_Alice,self).__init__()
        self.main = nn.Sequential(
            nn.Linear(n_input,n_input),
            nn.Sigmoid(),
            nn.Conv1d(n_input,2,(4,4),stride=1),
            nn.Sigmoid(),
            nn.Conv1d(2,(4,4),stride=2),
            nn.Sigmoid(),
            nn.Conv1d(n_input,2,(4,4),stride=1),
            nn.Sigmoid(),
            nn.Conv1d(n_input,2,(4,4),stride=1),
            nn.Tanh()
        )

    def forward(self,input):
        output = self.main(input)
        return output

