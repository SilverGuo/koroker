import os

import tensorflow as tf
import numpy as np

from .base import BaseSeqLabel
from .utils.data_io import embed_from_npy, load_pickle
from .utils.date_process import pad_batch, create_batch
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
        self.label_pred = None
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
            self.label_pred = tf.cast(tf.argmax(self.logits, axis=2), tf.int32)

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
                raise ValueError(
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
    def build_feed_dict(self, sample, label=None, dropout=None):
        # word id list
        word_id, seq_len = pad_batch(sample['word_id'], 0)

        # build feed dict
        feed = {
            self.word_id: word_id,
            self.seq_len: seq_len
        }

        # if has char lstm
        if self.config.lstm_char:
            char_id, word_len = pad_batch(sample['char_id'], 0, 'char')
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

    # prepare data for train
    def prepare_data(self):
        data = load_pickle(self.config.data_path)
        return data['train'], data['dev'], data['test']

    # run epoch
    def run_epoch(self, sess, train, dev, epoch):
        # number of batch
        num_batch = (len(train[1])+self.config.batch_size-1) // self.config.batch_size

        for idx, (sample, label) in enumerate(create_batch(train,
                                                           self.config.batch_size,
                                                           self.config.has_char)):

            feed_dict, _ = self.build_feed_dict(sample, label)

            _, train_loss, summary = \
                sess.run([self.train_op, self.loss, self.summary_merge],
                         feed_dict=feed_dict)

            if idx % 10 == 0:
                self.file_writer.add_summary(summary, epoch * num_batch + idx)
                print('epoch {} batch {}, train loss {:04.2f}'.format(epoch,
                                                                      idx,
                                                                      train_loss))

        accuracy, f1 = self.run_evaluate(sess, dev)
        self.logger.info("dev acc {:04.2f}, f1 {:04.2f}".format(100*accuracy,
                                                                100*f1))

        return accuracy, f1

    # batch predict
    def predict_batch(self, sess, sample):
        # get the feed dict
        feed_dict, seq_len = self.build_feed_dict(sample)

        if self.config.has_crf:
            seq_viterbi = []
            logits, transition_param = sess.run([self.logits, self.transition_param],
                                                feed_dict=feed_dict)
            # viterbi loop
            for logit, lseq in zip(logits, seq_len):
                # seq mask
                logit = logit[:lseq]
                viterbi, _ = tf.contrib.crf.viterbi_decode(logit,
                                                           transition_param)
                seq_viterbi.append(viterbi)

            return seq_viterbi, seq_len

        else:
            # prediction without crf
            label_pred = sess.run(self.label_pred, feed_dict=feed_dict)

            return label_pred, seq_len

    # run evaluate
    def run_evaluate(self, sess, data):
        # accuracy sequence
        seq_accuracy = []

        for sample, label in create_batch(data, self.config.batch_size):
            label_pred, seq_len = self.predict_batch(sess, sample)

            for y, y_pred, lseq in zip(label, label_pred, seq_len):
                # mask
                y = y[:lseq]
                y_pred = y_pred[:lseq]

                seq_accuracy += [r == p for (r, p) in zip(y, y_pred)]

        accuracy = np.mean(seq_accuracy)
        # if no f1 defined
        f1 = accuracy
        return accuracy, f1

    # train model
    def train(self):

        saver = tf.train.Saver()

        # early stop
        best_score = 0.0
        num_epoch_no_improve = 0

        # load and split data
        train, dev, test = self.prepare_data()

        with tf.Session() as sess:
            sess.run(self.var_init)

            # load trained model
            if self.config.load_model:
                self.logger.info('load trained model from '+self.config.trained_model_path)
                saver.restore(sess, self.config.trained_model_path)

            # tensorboard
            self.add_summary(sess)

            # train epoch
            for epoch in range(self.config.num_epoch):
                # epoch start
                self.logger.info('epoch {} out of {}'.format(epoch + 1,
                                                             self.config.num_epoch))

                # run epoch
                accuracy, f1 = self.run_epoch(sess, train, dev, epoch)

                # learning rate decay
                self.config.lr *= self.config.lr_decay

                # early stop
                if f1 >= best_score:
                    num_epoch_no_improve = 0
                    if not os.path.exists(self.config.model_dir):
                        os.makedirs(self.config.model_dir)
                    saver.save(sess,
                               os.path.join(self.config.model_dir, 'alpha'),
                               global_step=epoch)
                    best_score = f1
                    self.logger.info('new best score {:04.2f}'.format(best_score))

                else:
                    num_epoch_no_improve += 1
                    if num_epoch_no_improve > self.config.early_stop:
                        self.logger.info('early stop {} without improvement'.format(num_epoch_no_improve))

                        # stop train
                        break

    # need to implement
    def analysis(self, doc):
        super(ModelLstmCrf, self).analysis(doc)
