import numpy as np
import torch
from data.config import MSG_LEN, KEY_LEN, CHAR_SET

def generate_sequence(n):
    seq = np.random.choice(CHAR_SET, n) #generate random n-bit long sequences
    seq = torch.from_numpy(seq.reshape(n,1))
    return seq

def get_data():
    plain = generate_sequence(MSG_LEN)
    key = generate_sequence(KEY_LEN)
    return (plain, key)
