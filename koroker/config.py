import os

from .base import BaseConfig
from .utils.logger import new_logger


class ConfigPrep(BaseConfig):

    def __init__(self, config_path):
        # parser config
        super(ConfigPrep, self).__init__(config_path)

        # io
        self.train_path = self.config['io']['train']
        self.dev_path = self.config['io']['dev']
        self.test_path = self.config['io']['test']
        self.file_format = self.config['io']['format']
        output_path = self.config['io']['output']
        self.data_path = os.path.join(output_path, 'data.pkl')
        self.vocab_path = os.path.join(output_path, 'vocab.pkl')

        # embed
        self.use_word = self.config['embed'].getboolean('use_word', True)
        self.use_char = self.config['embed'].getboolean('use_char', False)
        self.word_in_path = self.config['embed'].get('word_embed', '/fake_path')
        self.char_in_path = self.config['embed'].get('char_embed', '/fake_path')
        self.word_out_path = os.path.join(output_path, 'word_embed.npy')
        self.char_out_path = os.path.join(output_path, 'char_embed.npy')

        # vocab
        self.max_word_vocab = self.config['vocab'].getint('max_word', 25000)
        self.max_char_vocab = self.config['vocab'].getint('max_char', 100)

        # text process
        self.lower_word = self.config['text'].getboolean('lower_word', True)


class ConfigLstmCrf(BaseConfig):

    def __init__(self, config_path):
        # parser config
        super(ConfigLstmCrf, self).__init__(config_path)

        # io
        input_path = self.config['io'].get('input', '/fake_path')
        self.data_path = os.path.join(input_path, 'data.pkl')
        self.vocab_path = os.path.join(input_path, 'vocab.pkl')
        output_dir = self.config['io']['output']
        self.model_dir = os.path.join(output_dir, 'model/')
        self.summary_dir = os.path.join(output_dir, 'summary/')
        self.log_path = os.path.join(output_dir, 'log.txt')

        # embed
        self.word_embed_path = os.path.join(input_path, 'word_embed.npy')
        self.char_embed_path = os.path.join(input_path, 'char_embed.npy')
        self.use_word_embed = self.config['embed'].getboolean('use_word', True)
        self.use_char_embed = self.config['embed'].getboolean('use_char', False)
        self.word_embed_dim = self.config['embed'].getint('word_dim', 300)
        self.char_embed_dim = self.config['embed'].getint('char_dim', 100)

        # checkpoint
        self.trained_model_path = self.config['checkpoint'].get('model', False)

        # computation graph
        self.has_char = self.config['graph'].getboolean('char', True)
        self.has_crf = self.config['graph'].getboolean('crf', True)

        # train parameter
        self.train_embed = self.config['train'].getboolean('embed', True)
        self.num_epoch = self.config['train'].getint('epoch', 15)
        self.dropout = self.config['train'].getfloat('dropout', 0.5)
        self.batch_size = self.config['train'].getint('batch', 20)
        self.lr = self.config['train'].getfloat('lr', 0.001)
        self.lr_method = self.config['train'].get('opt', 'adam')
        self.lr_decay = self.config['train'].getfloat('decay', 1.0)
        self.early_stop = self.config['train'].getint('early', 3)

        # hyper parameter
        self.word_lstm_hidden = self.config['hyper'].getint('word_hidden', 300)
        self.char_lstm_hidden = self.config['hyper'].getint('char_hidden', 100)
        self.grad_clip = self.config['hyper'].getfloat('grad_clip', 5.0)

        # log
        self.logger = new_logger('train_alpha', self.log_path)
