import torch
from torch.optim import optim
from argparse import ArgumentParser
from src.model import AdversarialNeuralCryptoNet
from src.config import *
from src.data_loader import train_loader
from src.data_loader import test_loader

def get_parser():
    parser = ArgumentParser(description='Pytorch Implementation of Adversarial Neural Cryptography')

    parser.add_argument('--msg-len', type=int,
                        dest='msg_len', help='message length',
                        metavar='MSG_LEN', default=MSG_LEN)
    parser.add_argument('-lr','--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='Number of Epochs in Adversarial Training',
                        metavar='EPOCHS', default=NUM_EPOCHS)
    parser.add_argument('-bs','--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)
    parser.add_argument('--no-cuda',action='store_true', default=NO_CUDA,dest='no_cuda',
                    help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, metavar=RANDOM_SEED,dest='seed',
                    help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=LOG_INTERVAL, metavar=LOG_INTERVAL,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
    return parser

def train(model):
    model.train()

def test(model):
    model.eval()


def main():
    parser = get_parser()
    args = parser.parse_args()

    no_cuda = args.no_cuda
    msg_len=args.msg_len
    epochs=args.epochs
    batch_size=args.batch_size
    learning_rate=args.learning_rate
    seed = args.seed
    momentum = args.momentum


    use_cuda = not no_cuda and torch.cuda.is_available()

    if use_cuda:
        torch.cuda.manual_seed(seed)
        kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        torch.manual_seed(seed)
        kwargs = {}

    train_loader = train_loader()
    test_loader = test_loader()

    model = AdversarialNeuralCryptoNet()
    if use_cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(),lr = learning_rate, momentum = momentum)

    for epoch in range(1,epochs+1):
        train(model,epoch)
        test(model)

if __name__ == '__main__':
    main()
