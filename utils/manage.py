import torch
from torch.autograd import Variable
from models.CryptoNet import ABCryptoNet
from models.CryptoNet import ECryptoNet
import torch.optim as optim
import tensorflow as tf


class Manager(object):
    def __init__(self, args, texts):
        self.msg_len = args.msg_len
        self.no_cuda = args.no_cuda
        self.epochs = args.epochs
        self.lr = args.lr
        self.seed = args.seed
        self.display = args.display
        self.max_iter = 1
        (self.plain_text, self.msg_key) = texts
        self.build_model()
        self.build_optimizer()

    def concat_input(self,in1, in2, name):
        _in = torch.cat((self.plain_text,self.msg_key),2).float() #Axis = 1
        _in = Variable(_in, requires_grad = True)#.unsqueeze(1)
        print('Created Input for {}: {}.'.format(name,_in.size()))
        return _in

    def build_optimizer(self):
        self.alice_optimizer = optim.Adam(self.alice_net.parameters(), lr=self.lr)
        self.bob_optimizer = optim.Adam(self.bob_net.parameters(), lr=self.lr)
        self.eve_optimizer = optim.Adam(self.eve_net.parameters(), lr=self.lr)

    def build_model(self):
        """ Build Three CryptoNet Models """

        self.alice_net = ABCryptoNet()
        self.bob_net = ABCryptoNet()
        self.eve_net = ECryptoNet()
        print(self.alice_net)
        print(self.bob_net)
        print(self.eve_net)

        use_cuda = not self.no_cuda and torch.cuda.is_available()
        if use_cuda:
            self.alice_net.cuda()
            self.bob_net.cuda()
            self.eve_net.cuda()

    def reconstruction_loss(self,msg, output):
        """Reconstruction error."""
        r_loss = torch.abs(msg - output).mean() / 2
        return r_loss

    def tf_reconstruction_loss(self,msg, output):
        """Reconstruction error."""
        r_loss = tf.reduce_mean(tf.abs(tf.subtract(msg, output))) / 2
        return r_loss

    def bits_loss(self,msg, output, message_length):
        """Reconstruction error in number of different bits."""
        return reconstruction_loss(msg, output) * message_length

    def train_ab(self, _in):
        print("\tTraining Alice and Bob for {} iterations...".format(self.max_iter))
        in_bob = self.concat_input(_in,self.msg_key, 'Bob') #create input for bob (CipherText+msgkey)
        for i in range(self.max_iter):
            in_bob = self.bob_net(in_bob)

    def train_e(self, _in):
        print("\tTraining Alice and Bob for {} iterations...".format(self.max_iter))
        for i in range(self.max_iter):
            bob_output = self.eve_net(in_bob)

    def train(self):
        ''' Train the model'''
        in_alice = self.concat_input(self.plain_text,self.msg_key, 'Alice')
        alice_output = self.alice_net(in_alice) # create cipher text
        for epoch in range(1,self.epochs+1):
            self.train_ab(alice_output)
            eve_output = self.eve_net(alice_output)
            import pdb; pdb.set_trace()
            eve_loss = self.reconstruction_loss(self.plain_text.float(), eve_output.data)
            eve_loss = self.tf_reconstruction_loss(self.plain_text.data, eve_output)

            bob_reconst_loss = self.reconstruction_loss(self.plain_text, bob_output)
            bob_loss = bob_reconst_loss + (0.5 - eve_loss) ** 2

            eve_bit_loss = bits_loss(self.plain_text, eve_output, self.msg_len)
            bob_bit_loss = bits_loss(self.plain_text, bob_output, self.msg_len)

            if epoch % self.display:
                pass
