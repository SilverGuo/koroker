from config import ConfigPrep
from util_sys import save_pickle
from util_data import DataNER, conll_norm, \
                      load_embed, generate_embed, \
                      generate_vocab

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
    return ConfigPrep(train_path=config['input']['train'], 
                      dev_path=config['input']['dev'], 
                      test_path=config['input']['test'], 
                      output_dir=config['output']['dir'], 
                      word_lower=config['text'].getboolean('lower'), 
                      norm_digit=config['text'].getboolean('digit'), 
                      word_embed_path=config['embed']['word'], 
                      num_word_embed=config['embed'].getint('num'), 
                      word_vocab=config['vocab']['word'], 
                      char_vocab=config['vocab']['char'])
    

# preprocess
def preprocess(config):

    # load dataset
    train_set = DataNER(config.train_path, config, conll_norm)
    dev_set = DataNER(config.dev_path, config, conll_norm)
    test_set = DataNER(config.test_path, config, conll_norm)

    # create vocab
    vword, vchar = train_set.create_vocab()
    # create entity dict
    entity_dict = train_set.create_entity()

    # load embed
    if config.word_embed_path != '':
        dembed = load_embed(config.word_embed_path)
        # union vocab
        vword = list(vword.union(list(dembed.keys())[:config.num_word_embed]))
        # embed generate
        wembed = generate_embed(vword, dembed)
        """
        TO BE DONE
        """

    else:
        # fix order
        vword = list(vword)
        vchar = list(vchar)
    
    # temp solution for train step
    print(len(vword))
    print(len(vchar))
    print(len(entity_dict))
    
    # generate vocab mapping
    word_idx, idx_word = generate_vocab(vword)
    char_idx, idx_char = generate_vocab(vchar)

    # generate dataset
    train_set.generate_dataset(word_idx, char_idx, entity_dict)
    dev_set.generate_dataset(word_idx, char_idx, entity_dict)
    test_set.generate_dataset(word_idx, char_idx, entity_dict)

    # output
    save_pickle(train_set.dataset, os.path.join(config.data_dir + 'train.pkl'))
    save_pickle(dev_set.dataset, os.path.join(config.data_dir + 'dev.pkl'))
    save_pickle(test_set.dataset, os.path.join(config.data_dir + 'test.pkl'))
    save_pickle(entity_dict, config.entity_path)

    return


if __name__ == '__main__':
    config = config_parse(args.config)

    preprocess(config)

    
    
    

