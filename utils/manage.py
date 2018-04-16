import torch
from torch.autograd import Variable
from models.CryptoNet import ABCryptoNet
from models.CryptoNet import ECryptoNet
from data.config import MSG_LEN, KEY_LEN


class Manager(object):
    def __init__(self, args, texts):
        self.no_cuda = args.no_cuda
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.seed = args.seed
        self.momentum = args.momentum
        (self.plain_text, self.msg_key) = texts
        self.build_model()

    def concat_input(self,in1, in2, name):
        _in = torch.cat((self.plain_text,self.msg_key),2).float() #Axis = 1
        _in = Variable(_in, requires_grad = True)#.unsqueeze(1)
        print('Created Input for {}: {}.'.format(name,_in.size()))
        return _in

    def losses(self):
        pass

    def build_model(self):
        """ Build Three CryptoNet Models """

        self.aliceNet = ABCryptoNet()
        self.bobNet = ABCryptoNet()
        self.eveNet = ECryptoNet()
        print(self.aliceNet)
        print(self.bobNet)
        print(self.eveNet)

        use_cuda = not self.no_cuda and torch.cuda.is_available()
        if use_cuda:
            self.aliceNet.cuda()
            self.bobNet.cuda()
            self.eveNet.cuda()
        self.in_alice = self.concat_input(self.plain_text,self.msg_key, 'Alice')

    def get_loss(self, name = None):
        if name == 'eve':
            pass
        elif name == 'bob':
            pass
        else:
            Print('Invalid Net(name) Passed.')

    def train(self):
        ''' Train the model'''
        # import pdb; pdb.set_trace()
        for epoch in range(1,self.epochs+1):
            cipher_text = self.aliceNet(self.in_alice) # create cipher text
            in_bob = self.concat_input(cipher_text,self.msg_key, 'Bob') #create input for bob (CipherText+msgkey)
            self.plain_text_bob = self.bobNet(in_bob)
            self.plain_text_eve = self.eveNet(cipher_text)
