import os
import pickle

import numpy as np


# save to pickle
def save_pickle(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(data, f)


# load pickle
def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


# numpy matrix load
def embed_from_npy(embed_path):
    return np.load(embed_path)
