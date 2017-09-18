import logging
import pickle
import os
import numpy as np

LOG_FORMAT = '%(asctime)s:%(levelname)s;%(message)s'

# create logger and file handler
def create_logger(log_name, log_path):

    # create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    # config for log system
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)

    # create file handler
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

    logging.getLogger().addHandler(handler)
    return logger

# numpy matrix load
def embed_from_npy(embed_path):
    if embed_path is not None:
        return np.load(embed_path)
    else:
        return None

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


