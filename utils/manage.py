import torch
from torch.autograd import Variable
from models.CryptoNet import ABCryptoNet
from models.CryptoNet import ECryptoNet

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

    def build_model(self):
        """ Build Three CryptoNet Models """
        self.aliceNet = ABCryptoNet()
        self.bobNet = ABCryptoNet()
        self.eveNet = ECryptoNet()
        print(self.aliceNet)

        use_cuda = not self.no_cuda and torch.cuda.is_available()
        if use_cuda:
            self.aliceNet.cuda()
            self.bobNet.cuda()
            self.eveNet.cuda()
        _in = torch.cat((self.plain_text,self.msg_key),0).float()
        print('Input to Alice {}.'.format(_in.size()))
        self.in_alice = Variable(_in, requires_grad = True).unsqueeze(1)

    def train(self):
        ''' Train the model'''
        for epoch in range(1,self.epochs+1):
            out_alice = self.aliceNet(self.in_alice)
            out_bob = self.bobNet(out_alice)
            out_eve = self.eveNet(out_alice)
            import pdb;pdb.set_trace()
