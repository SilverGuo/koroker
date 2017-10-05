import tensorflow as tf

from .base import BaseSeqLabel
from .utils.pipeline import embed_from_npy, load_pickle
from .utils.date_process import pad_batch
from .config import ConfigLstmCrf


# lstm crf model for sequence labeling
class ModelLstmCrf(BaseSeqLabel):

    def __init__(self, config_path):
        # config class
        self.config = ConfigLstmCrf(config_path)

        # embed
        self.word_embed = None
        self.char_embed = None
        self.load_embed()

        # vocab and label dict
        self.word_dict = None
        self.char_dict = None
        self.label_dict = None
        self.load_vocab()

        # placeholder
        self.seq_len = None
        self.word_id = None
        self.word_len = None
        self.char_id = None
        self.label = None
        self.dropout = None

        # graph var
        self.char_lstm_output = None
        self.word_lstm_output = None
        self.logits = None
        self.pred_tag = None
        self.transition_param = None
        self.loss = None
        self.train_op = None
        self.var_init = None
        self.summary_merge = None
        self.file_writer = None
        self.build_graph()

    # load embed matrix
    def load_embed(self):
        # word embed
        if self.config.use_word_embed:
            self.word_embed = embed_from_npy(self.config.word_embed_path)
            self.config.word_embed_dim = self.word_embed.shape[-1]

        # char embed
        if self.config.use_char_embed:
            self.char_embed = embed_from_npy(self.config.char_embed_path)
            self.config.char_embed_dim = self.char_embed.shape[-1]

    # load vocab
    def load_vocab(self):
        vocab_dict = load_pickle(self.config.vocab_path)

        # sample dict
        self.word_dict = vocab_dict['sample']['word']
        if self.config.has_char:
            self.char_dict = vocab_dict['sample']['char']

        # label dict
        self.label_dict = vocab_dict['label']

    # add placeholder
    def add_placeholder(self):
        # sequence length
        # (batch size)
        self.seq_len = tf.placeholder(tf.int32,
                                      shape=[self.config.batch_size],
                                      name='seq_len')

        # word id list
        # (batch size, max len in batch)
        self.word_id = tf.placeholder(tf.int32,
                                      shape=[self.config.batch_size, None],
                                      name='word_id')

        # word length
        # (batch size, max len in batch)
        self.word_len = tf.placeholder(tf.int32,
                                       shape=[self.config.batch_size, None],
                                       name='word_len')

        # char id list
        # (batch size, max len in batch, max length of word)
        self.char_id = tf.placeholder(tf.int32,
                                      shape=[self.config.batch_size, None, None],
                                      name='char_id')

        # label
        # (batch size, max len in batch)
        self.label = tf.placeholder(tf.int32,
                                    shape=[self.config.batch_size, None],
                                    name='label')

        # hyper param
        self.dropout = tf.placeholder(tf.float32, shape=[],
                                      name='dropout')

    # add char lstm
    def add_char_lstm(self):
        with tf.variable_scope('char_embed'):
            # use pretrained embed
            if self.config.use_char_embed:
                char_embed_mat = tf.Variable(self.char_embed,
                                             name='matrix',
                                             dtype=tf.float32,
                                             trainable=self.config.train_embed)
            else:
                char_embed_mat = tf.get_variable(name='matrix',
                                                 dtype=tf.float32,
                                                 shape=[len(self.char_dict),
                                                        self.config.char_embed_dim])

            # lookup
            char_rep = tf.nn.embedding_lookup(char_embed_mat, self.char_id,
                                              name='representation')

        with tf.variable_scope('char_lstm'):
            # reshape before lstm
            shape_op = tf.shape(char_rep)
            char_rep = tf.reshape(char_rep,
                                  shape=[-1, shape_op[2], self.config.char_embed_dim])
            word_len = tf.reshape(self.word_len, shape=[-1])

            # fw and bw layer
            lstm_fw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_hidden,
                                              state_is_tuple=True)
            lstm_bw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_hidden,
                                              state_is_tuple=True)

            # bi lstm for char
            _, ((_, final_fw), (_, final_bw)) = \
                tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw,
                                                char_rep,
                                                sequence_length=word_len,
                                                dtype=tf.float32)

            # concate final state in bi-direction
            final_state = tf.concat([final_fw, final_bw], axis=1)

        self.char_lstm_output = tf.reshape(final_state,
                                           shape=[-1, shape_op[1],
                                                  2*self.config.char_lstm_hidden])

    # add word lstm
    def add_word_lstm(self):
        # word embed
        with tf.variable_scope('word_embed'):
            # use pretrained embed
            if self.config.use_word_embed:
                word_embed_mat = tf.Variable(self.word_embed,
                                             name='matrix',
                                             dtype=tf.float32,
                                             trainable=self.config.train_embed)
            else:
                word_embed_mat = tf.get_variable(name='matrix',
                                                 dtype=tf.float32,
                                                 shape=[len(self.word_dict),
                                                        self.config.word_embed_dim])

            # lookup
            word_rep = tf.nn.embedding_lookup(word_embed_mat, self.word_id,
                                              name='representation')

        if self.config.has_char:
            # concate with word embed
            word_rep = tf.concat([word_rep, self.char_lstm_output], axis=2)

        # dropout before word lstm
        word_rep = tf.nn.dropout(word_rep, self.dropout)

        # bi lstm for word
        with tf.variable_scope('word_lstm'):
            lstm_fw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_hidden,
                                              state_is_tuple=True)
            lstm_bw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_hidden,
                                              state_is_tuple=True)
            (output_fw, output_bw), _ = \
                tf.nn.bidirectional_dynamic_rnn(lstm_fw,
                                                lstm_bw,
                                                word_rep,
                                                sequence_length=self.seq_len,
                                                dtype=tf.float32)

            output = tf.concat([output_fw, output_bw], axis=2)

        # word lstm output
        self.word_lstm_output = tf.nn.dropout(output, self.dropout)

    # add logits op
    def add_logits_op(self):
        # logits
        with tf.variable_scope('logits'):
            w_logits = tf.get_variable('weight_logits',
                                       shape=[2 * self.config.word_lstm_hidden,
                                              len(self.label_dict)],
                                       dtype=tf.float32)
            b_logits = tf.get_variable('bias_logits',
                                       shape=[len(self.label_dict)],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer())

            num_step = tf.shape(self.word_lstm_output)[1]
            word_vec = tf.reshape(self.word_lstm_output,
                                  [-1, 2*self.config.word_lstm_hidden])

            # linear
            pred_vec = tf.matmul(word_vec, w_logits) + b_logits

            self.logits = tf.reshape(pred_vec,
                                     [-1, num_step, len(self.label_dict)])

    # add pred without crf
    def add_pred_no_crf(self):
        if not self.config.has_crf:
            self.pred_tag = tf.cast(tf.argmax(self.logits, axis=2), tf.int32)

    # add loss op
    def add_loss_op(self):
        # with crf
        if self.config.has_crf:
            log_ll, self.transition_param = \
                tf.contrib.crf.crf_log_likelihood(self.logits, self.label,
                                                  self.seq_len)
            self.loss = tf.reduce_mean(-log_ll)

        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits,
                                                                  self.label)
            mask = tf.sequence_mask(self.seq_len)
            loss_mask = tf.boolean_mask(loss, mask)
            self.loss = tf.reduce_mean(loss_mask)

        # loss into summary
        tf.summary.scalar('loss', self.loss)

    # add train op
    def add_train_op(self):
        # optimizer
        optimizer = None
        with tf.variable_scope('optimizer'):
            # lr method
            if self.config.lr_method == 'adam':
                optimizer = tf.train.AdamOptimizer(self.config.lr)
            elif self.config.lr_method == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.config.lr)
            elif self.config.lr_method == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
            elif self.config.lr_method == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.config.lr)
            else:
                raise NotImplementedError(
                    'Unknown optimizer {}'.format(self.config.lr_method))

        # gradient clip
        if self.config.grad_clip > 0:
            grad_op, var_op = zip(*optimizer.compute_gradients(self.loss))
            grad_op, _ = tf.clip_by_global_norm(grad_op, self.config.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(grad_op, var_op))
        else:
            self.train_op = optimizer.minimize(self.loss)

    # add init
    def add_init_op(self):
        self.var_init = tf.global_variables_initializer()

    # add summary
    def add_summary(self, sess):
        self.summary_merge = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.summary_dir,
                                                 sess.graph)

    # graph build
    def build_graph(self):
        self.add_placeholder()
        self.add_char_lstm()
        self.add_word_lstm()
        self.add_logits_op()
        self.add_pred_no_crf()
        self.add_loss_op()
        self.add_train_op()
        self.add_init_op()

    # feed dict for model
    def build_feed_dict(self, doc, label=None, dropout=None):
        # word id list
        word_id, seq_len = pad_batch(doc['word_id'], 0)

        # build feed dict
        feed = {
            self.word_id: word_id,
            self.seq_len: seq_len
        }

        # if has char lstm
        if self.config.lstm_char:
            char_id, word_len = pad_batch(doc['char_id'], 0, 'char')
            feed[self.char_id] = char_id
            feed[self.word_len] = word_len

        if label is not None:
            label, _ = pad_batch(label, 0)
            feed[self.label] = label

        if dropout is not None:
            feed[self.dropout] = dropout
        else:
            feed[self.dropout] = self.config.dropout

        return feed, seq_len

    def analysis(self, doc):
        super(ModelLstmCrf, self).analysis(doc)
