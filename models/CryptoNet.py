import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class ABCryptoNet(nn.Module):
    '''
    Standard Convolutional layers setup used by Alice and Bob.
    Input: 2N*1d tensor, PlainText+MessageKey(Alice) or CipherText+Key(Bob)
    Output: N*1d tensor, CipherText(Alice) or PlainText(Bob)
    '''
    def __init__(self):
        super(ABCryptoNet,self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1,1),
            nn.Sigmoid(),
            nn.Conv1d(1,2,4,stride=1),
            nn.Sigmoid(),
            nn.Conv1d(2,4,2,stride=2),
            nn.Sigmoid(),
            nn.Conv1d(4,4,1,stride=1),
            nn.Sigmoid(),
            nn.Conv1d(4,1,1,stride=1),
            nn.Tanh()
        )
    def forward(self,input):
        output = self.net(input)
        return output

class ECryptoNet(nn.Module):
    '''
    Standard Convolutional layers setup used by Eve.
    Input: N*1d tensor, CipherText
    Output: N*1d tensor, predicted PlainText
    '''
    def __init__(self):
        super(ECryptoNet,self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1,2),
            nn.Sigmoid(),
            nn.Conv1d(1,2,4,stride=1),
            nn.Sigmoid(),
            nn.Conv1d(2,4,2,stride=2),
            nn.Sigmoid(),
            nn.Conv1d(4,4,1,stride=1),
            nn.Sigmoid(),
            nn.Conv1d(4,1,1,stride=1),
            nn.Tanh()
        )

    def forward(self,input):
        output = self.net(input)
        return output

