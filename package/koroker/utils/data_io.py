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


# save numpy
def embed_to_npy(data, embed_path):
    if not os.path.exists(os.path.dirname(embed_path)):
        os.makedirs(os.path.dirname(embed_path))
    np.save(embed_path, data)
    return


# numpy matrix load
def embed_from_npy(embed_path):
    return np.load(embed_path)


# parse conll data
def read_conll(data_path):
    sample_word, label = [], []
    tok, tag = [], []

    # read file
    with open(data_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line == '' or line.startswith('-DOCSTART-'):
                if tok != [] and tag != []:
                    sample_word.append(tok)
                    label.append(tag)
                    tok, tag = [], []
            else:
                temp = line.split()
                tok.append(temp[0])
                tag.append(temp[3])

    # if last sentence
    if tok != [] and tag != []:
        sample_word.append(tok)
        label.append(tag)

    # sample and label
    return sample_word, label
