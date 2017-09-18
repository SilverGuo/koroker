from config import ConfigNER
from model import DeepNER
from util_sys import load_pickle

import os
import argparse
import configparser


# arg parse
parser = argparse.ArgumentParser(description='config file')
parser.add_argument('--config', type=str, 
                    help='config file path')
args = parser.parse_args()


# config parse
def config_parse(config_path):
    # read ini file
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_path)

    # create config
    return ConfigNER(train_path=config['input']['train'], 
                     dev_path=config['input']['dev'], 
                     test_path=config['input']['test'],
                     entity_path=config['input']['entity'],
                     model_path=config['trained']['model'], 
                     output_dir=config['output']['dir'])
    

# train model
def train_model(config):

    # build model
    model = DeepNER(config)
    model.build_graph()

    # load data
    train = load_pickle(config.train_path)
    dev = load_pickle(config.dev_path)
    test = load_pickle(config.test_path)

    # entity dict
    entity_dict = load_pickle(config.entity_path)

    # train
    model.train(train, dev, entity_dict)
    return


if __name__ == '__main__':
    config = config_parse(args.config)
    # train
    train_model(config)
