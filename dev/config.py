from util_sys import create_logger

import os


class ConfigPrep:

    def __init__(self, 
                 train_path, 
                 dev_path, 
                 test_path, 
                 output_dir, 
                 word_lower=True, 
                 norm_digit=True, 
                 word_embed_path='',
                 num_word_embed=25000,  
                 word_vocab='', 
                 char_vocab=''):

        # input file
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path

        # output
        self.output_dir = output_dir
        self.data_dir = os.path.join(self.output_dir, 'data/')
        self.vocab_dir = os.path.join(self.output_dir, 'vocab/')
        self.entity_path = os.path.join(self.output_dir, 'entity.pkl')

        # text preprocess
        self.word_lower = word_lower
        self.norm_digit = norm_digit

        # word embedding
        self.word_embed_path = word_embed_path
        self.num_word_embed = num_word_embed

        # vocab
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab


class ConfigNER:

    def __init__(self, 
                 train_path, 
                 dev_path, 
                 test_path, 
                 entity_path, 
                 model_path, 
                 output_dir, 
                 lstm_char = True, 
                 num_tag=9, 
                 use_crf=True, 
                 word_vocab_size=20313,
                 char_vocab_size=59, 
                 word_dim = 300, 
                 char_dim = 100, 
                 word_embed_path='', 
                 char_embed_path='', 
                 load_model=False, 
                 train_embed=True, 
                 num_epoch=15, 
                 dropout=0.5, 
                 batch_size=20, 
                 lr_method='adam', 
                 lr=0.0005, 
                 lr_decay=0.8,
                 early_stop=3,  
                 char_hidden_size=100, 
                 word_hidden_size=300, 
                 grad_clip=5.0):

        # data set
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.entity_path = entity_path

        # trained model
        self.model_path = model_path

        # output
        self.output_dir = output_dir
        self.model_dir = os.path.join(self.output_dir, 'model/')
        self.summary_dir = os.path.join(self.output_dir, 'summary/')
        self.log_path = os.path.join(self.output_dir, 'log.txt')

        # network
        self.lstm_char = lstm_char
        self.num_tag = num_tag
        self.use_crf = use_crf
        
        # embedding
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.word_dim = word_dim
        self.char_dim = char_dim
        self.word_embed_path = word_embed_path
        self.char_embed_path = char_embed_path

        # training
        self.load_model = load_model
        self.train_embed = train_embed
        self.num_epoch = num_epoch
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr_method = lr_method
        self.lr = lr
        self.lr_decay = lr_decay
        self.early_stop = early_stop

        # model hyperparam
        self.char_hidden_size = char_hidden_size
        self.word_hidden_size = word_hidden_size
        self.grad_clip = grad_clip

        # initial output dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # initial logger
        self.logger = create_logger('train_alpha', self.log_path)

        return
        


