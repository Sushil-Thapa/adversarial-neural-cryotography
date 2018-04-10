import torch
import torch.optim as optim
from argparse import ArgumentParser
from data.config import *
from utils.data_loader import get_data
from utils.manage import Manager

torch.manual_seed(42)
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
    parser.add_argument('--seed', type=int, default=MANUAL_SEED, metavar=MANUAL_SEED,dest='seed',
                    help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=LOG_INTERVAL, metavar=LOG_INTERVAL,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    inputs = get_data() # Generates and Loads data for training
    manager = Manager(args,inputs)
    manager.train()

if __name__ == '__main__':
    main()
