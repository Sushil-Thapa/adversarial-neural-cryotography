import numpy as np
import torch
from data.config import MSG_LEN, KEY_LEN, CHAR_SET, BATCH_SIZE

def generate_sequence(n_batch, n_char):
    seq = np.random.choice(CHAR_SET, n_char*n_batch) #generate random n-bit long sequences
    seq = torch.from_numpy(seq.reshape(n_batch,1,n_char))
    return seq

def get_data():
    plain = generate_sequence(BATCH_SIZE, MSG_LEN)
    key = generate_sequence(BATCH_SIZE, KEY_LEN)
    return (plain, key)
