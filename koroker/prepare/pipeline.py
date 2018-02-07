from ..base import BasePrepare
from ..config import ConfigPrep
from .data_set import DataNer, OutVoc
from ..utils.data_io import read_conll, embed_to_npy,\
 save_pickle, find_entities, filter_test
from ..utils.data_process import create_vocab, create_label, \
    load_embed, prepare_embed, vocab_mapping


class PrepareNer(BasePrepare):

    def __init__(self, config_path):
        self.config = ConfigPrep(config_path)

        self.train, self.dev, self.test = self.load_data()

        self.oov_set = self.build_oov()
        self.vocab_dict = self.process_data()

        # output data and vocab
        self.save_data()

    def load_data(self):
        if self.config.file_format == 'conll':
            read_file = read_conll
        elif self.config.file_format == "conll2":
            read_file = read_conll_2col
        else:
            read_file = read_conll

        return DataNer(self.config.train_path, read_file), \
            DataNer(self.config.dev_path, read_file), \
            DataNer(self.config.test_path, read_file)

    def process_data(self):
        # vocab from data set
        word_vocab, char_vocab = create_vocab(self.train.sample_tok
                                              + self.dev.sample_tok
                                              + self.test.sample_tok,
                                              self.config.max_word_vocab,
                                              self.config.max_char_vocab,
                                              self.config.lower_word)
        label_dict = create_label(self.train.label)

        # embedding
        # word
        if self.config.use_word:
            word_embed = load_embed(self.config.word_in_path)
            word_vocab = list(word_vocab.intersection(list(word_embed.keys())))
            # prepare embed
            word_embed = prepare_embed(word_vocab, word_embed)
            embed_to_npy(word_embed, self.config.word_out_path)
        else:
            # fix order
            word_vocab = list(word_vocab)
        # char
        if self.config.use_char:
            char_embed = load_embed(self.config.char_in_path)
            char_vocab = list(char_vocab.intersection(list(char_embed.keys())))
            # prepare embed
            char_embed = prepare_embed(char_vocab, char_embed)
            embed_to_npy(char_embed, self.config.char_out_path)
        else:
            # fix order
            char_vocab = list(char_vocab)

        # vocab mapping
        word_idx, _ = vocab_mapping(word_vocab)
        char_idx, _ = vocab_mapping(char_vocab)

        # data for training
        self.train.vocab_lookup(word_idx, char_idx, label_dict, self.config.lower_word)
        self.dev.vocab_lookup(word_idx, char_idx, label_dict, self.config.lower_word)
        self.test.vocab_lookup(word_idx, char_idx, label_dict, self.config.lower_word)
        self.oov_set.vocab_lookup(word_idx, char_idx, label_dict, self.config.lower_word)

        # vocab
        vocab_dict = dict()
        vocab_dict['sample'] = dict()
        vocab_dict['sample']['word'] = word_idx
        vocab_dict['sample']['char'] = char_idx
        vocab_dict['label'] = label_dict

        return vocab_dict

    def build_oov(self):
        train_entities = find_entities(self.train.sample_tok, self.train.label)
        oov_set = filter_test(self.dev.sample_tok, self.dev.label, set(train_entities))
        oov_set = OutVoc(oov_set)
        return oov_set


    def save_data(self):

        data = dict()
        data['train'] = self.train.data
        data['dev'] = self.dev.data
        data['test'] = self.test.data
        data['oov'] = self.oov_set.data

        save_pickle(data, self.config.data_path)
        save_pickle(self.vocab_dict, self.config.vocab_path)
