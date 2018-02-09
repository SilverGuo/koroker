import os
import pickle

import numpy as np

from itertools import chain

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
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


def filter_test(test_words, test_labels, overlap):
    """remove a sentence from development set if either:
        - it contains no entities 
        - the word of its entity is already present in overlap"""
    oov_set = []
    for words, labels in zip(test_words, test_labels):
        append_sent = False
        for word, label in zip(words, labels):
            if label != 'O':
                append_sent = True
                if word in overlap:
                    append_sent = False
                    
        if append_sent:
            oov_set.append((words,labels))
    return oov_set

def flatten(liste):
    return list(chain.from_iterable(liste))

def find_entities(words, labels):
    """detect entities in tagged conll corpus
        args: 
            words: list of tokenized sentences
            labels: list of associated labels
        returns:
        a set of merged entities found in corpus
    """
    entities = []
    sent_entities = []
    for words,labels in zip(words, labels):
        ent = []
        for i, (word, label) in enumerate(zip(words, labels)):
            if labels[i] != "O" and len(ent)==0: # start entities
                ent.append(words[i])
            elif labels[i] != "O" and len(ent)!=0: #continue entities
                ent.append(words[i])
            elif labels[i] == "O" and len(ent)!=0: # finish entities
                #ent = " ".join(ent) # merge lists to get unique entities
                entities.append(ent)
                ent = []
            else: 
                continue
            
    return flatten(entities)

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
                assert len(temp)==4,\
                "Each line should contain 4 elements, \n\
                if you have just current word and label use 'read_conll_2col' reading format"
                tok.append(temp[0])
                tag.append(temp[3])

    # if last sentence
    if tok != [] and tag != []:
        sample_word.append(tok)
        label.append(tag)

    # sample and label
    return sample_word, label



def read_conll_2col(data_path):
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
                tag.append(temp[1])
                assert(len(tok) == len(tag)),\
                "Different number of tags({}) and tokens ({}) in line: '{}'".format(len(tok), len(tag),line)
    
    # if last sentence
    if tok != [] and tag != []:
        sample_word.append(tok)
        label.append(tag)

    # sample and label
    assert len(sample_word) == len(label), \
    "different number of sentences and labels sentences"
    return sample_word, label


