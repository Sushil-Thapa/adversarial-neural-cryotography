import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from data.config import MSG_LEN, KEY_LEN


class ABCryptoNet(nn.Module):
    '''
    Standard Convolutional layers setup used by Alice and Bob.
    Input: 2N*1d tensor, PlainText+MessageKey(Alice) or CipherText+Key(Bob)
    Output: N*1d tensor, CipherText(Alice) or PlainText(Bob)
    '''
    def __init__(self):
        super(ABCryptoNet,self).__init__()
        self.fc_in =MSG_LEN + KEY_LEN
        self.fc_out = self.fc_in
        self.linear = nn.Linear(self.fc_in,self.fc_out)
        self.sigmoid = nn.Sigmoid()

        self.net = nn.Sequential(
            nn.Linear(self.fc_in,self.fc_out),
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
        # input_fc = input.view(-1, self.fc_in) #feed to FC
        # import pdb; pdb.set_trace()
        _out = self.linear(self.linear(input.squeeze(1)))
        output = self.net(_out.unsqueeze(1))
        print('one pass AB', output.size())
        return output

class ECryptoNet(nn.Module):
    '''
    Standard Convolutional layers setup used by Eve.
    Input: N*1d tensor, CipherText
    Output: N*1d tensor, predicted PlainText
    '''
    def __init__(self):
        super(ECryptoNet,self).__init__()
        self.fc_in =MSG_LEN
        self.fc_out = 2 * MSG_LEN
        self.linear = nn.Linear(self.fc_in,self.fc_out)
        self.sigmoid = nn.Sigmoid()
        self.net = nn.Sequential(
            nn.Conv1d(1,2,4,stride=1),
            nn.Sigmoid(),
            nn.Conv1d(2,4,2,stride=2),
            nn.Sigmoid(),
            nn.Conv1d(4,4,1,stride=1),
            nn.Sigmoid(),
            nn.Conv1d(4,1,1,stride=1),
            nn.Tanh()
        ) #eve stride [1,1,1,1] in github.com/rfratila/Adversarial-Neural-Cryptography/blob/master/net.py

    def forward(self,input):
        _out = self.linear(self.linear(input.squeeze(1)))
        output = self.net(_out.unsqueeze(1))
        print('one pass E',output.size())
        return output
