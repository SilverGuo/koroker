from config import ConfigNER
from util_sys import embed_from_npy
from util_data import pad_batch, create_batch, extract_chunk

import os
import numpy as np
import tensorflow as tf


class DeepNER(object):
   
    def __init__(self, config):
        # config class
        self.config = config

        # embed if use
        self.load_embed()

        # logger instantiated in config
        self.logger = self.config.logger
    
    # load embed matrix
    def load_embed(self):
        # word
        if self.config.word_embed_path == '':
            self.word_embed = None
        else:
            self.word_embed = embed_from_npy(self.config.word_embed_path)
        
        # char
        if self.config.char_embed_path == '':
            self.char_embed = None
        else:
            self.char_embed = embed_from_npy(self.config.char_embed_path)

    # add placeholder
    def add_placeholder(self):
        # sequence length 
        # (batch size)
        self.seq_len = tf.placeholder(tf.int32, 
                                      shape=[None], 
                                      name='seq_len')
        
        # word id list
        # (batch size, max len in batch)
        self.word_id = tf.placeholder(tf.int32, 
                                      shape=[None, None], 
                                      name='word_id')

        # word length
        # (batch size, max len in batch)
        self.word_len = tf.placeholder(tf.int32, 
                                       shape=[None, None], 
                                       name='word_len')
        
        # char id list
        # (batch size, max len in batch, max length of word)
        self.char_id = tf.placeholder(tf.int32, 
                                      shape=[None, None, None], 
                                      name='char_id')
        
        # label
        # (batch size, max len in batch)
        self.label = tf.placeholder(tf.int32, 
                                    shape=[None, None], 
                                    name='label')
        
        # hyper param
        self.dropout = tf.placeholder(tf.float32, shape=[], 
                                      name='dropout')
        self.lr = tf.placeholder(tf.float32, shape=[], 
                                 name='lr')
    
    # build feed dict
    def build_feed_dict(self, doc, label=None, 
                        lr=None, dropout=None):
        # use char lstm
        if self.config.lstm_char:
            word_id, seq_len = pad_batch(doc['word_id'], 0)
            char_id, word_len = pad_batch(doc['char_id'], 0, 'char')
        else:
            word_id, seq_len = pad_batch(doc['word_id'], 0)
        
        # build feed dict
        feed = {
            self.word_id: word_id, 
            self.seq_len: seq_len
        }

        if self.config.lstm_char:
            feed[self.char_id] = char_id
            feed[self.word_len] = word_len
        
        if label is not None:
            label, _ = pad_batch(label, 0)
            feed[self.label] = label
        
        if lr is not None:
            feed[self.lr] = lr
        else:
            feed[self.lr] = self.config.lr
        
        if dropout is not None:
            feed[self.dropout] = dropout
        else:
            feed[self.dropout] = self.config.dropout
        
        return feed, seq_len

    # add word embed op
    def add_word_embed_op(self):
        # word
        with tf.variable_scope('word'):
            # init dict
            if self.word_embed is None:
                word_dict = tf.get_variable(name='word_dict', 
                                            dtype=tf.float32, 
                                            shape=[self.config.word_vocab_size, 
                                                   self.config.word_dim])
            # pretrained embed
            else:
                word_dict = tf.Variable(self.word_embed, 
                                        name='word_dict', 
                                        dtype=tf.float32, 
                                        trainable=self.config.train_embed)
            # lookup
            word_rep = tf.nn.embedding_lookup(word_dict, self.word_id, 
                                              name='word_rep')
        # char
        with tf.variable_scope('char_lstm'):
            if self.config.lstm_char:
                # init dict
                if self.char_embed is None:
                    char_dict = tf.get_variable(name='char_dict', 
                                                dtype=tf.float32, 
                                                shape=[self.config.char_vocab_size, 
                                                       self.config.char_dim])
                # pretrained embed
                else:
                    char_dict = tf.Variable(self.char_embed, 
                                            name='char_dict', 
                                            dtype=tf.float32, 
                                            trainable=self.config.train_embed)
                # lookup
                # batch_size * word_len * char_len * char_dim
                char_rep = tf.nn.embedding_lookup(char_dict, self.char_id, 
                                                  name='char_rep')
                # reshape before lstm
                shape_op = tf.shape(char_rep)
                char_rep = tf.reshape(char_rep, 
                                      shape=[-1, shape_op[2], self.config.char_dim])
                word_len = tf.reshape(self.word_len, shape=[-1])

                # fw and bw layer
                lstm_fw = tf.contrib.rnn.LSTMCell(self.config.char_hidden_size, 
                                                  state_is_tuple=True)
                lstm_bw = tf.contrib.rnn.LSTMCell(self.config.char_hidden_size, 
                                                  state_is_tuple=True)
                
                # bi lstm for char
                _, ((_, final_fw), (_, final_bw)) = \
                    tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, 
                                                    char_rep, 
                                                    sequence_length=word_len, 
                                                    dtype=tf.float32)
                
                final_state = tf.concat([final_fw, final_bw], axis=1) 
                final_state = tf.reshape(final_state, 
                                         shape=[-1, shape_op[1], 
                                               2*self.config.char_hidden_size])

                # concate with word embed
                word_rep = tf.concat([word_rep, final_state], axis=2)

        # dropout before word lstm    
        word_rep = tf.nn.dropout(word_rep, self.dropout)

        # bi lstm for word
        with tf.variable_scope('word_lstm'):
            lstm_fw = tf.contrib.rnn.LSTMCell(self.config.word_hidden_size, 
                                              state_is_tuple=True)
            lstm_bw = tf.contrib.rnn.LSTMCell(self.config.word_hidden_size, 
                                              state_is_tuple=True)
            (output_fw, output_bw), _ = \
                tf.nn.bidirectional_dynamic_rnn(lstm_fw, 
                                                lstm_bw, 
                                                word_rep, 
                                                sequence_length=self.seq_len, 
                                                dtype=tf.float32)
            
            output = tf.concat([output_fw, output_bw], axis=2)
        
        # word lstm output
        self.lstm_output = tf.nn.dropout(output, self.dropout)
    
    # add logits op
    def add_logits_op(self):
        # logits
        with tf.variable_scope('logits'):
            W_logits = tf.get_variable('W_logits', 
                                       shape=[2*self.config.word_hidden_size, 
                                              self.config.num_tag], 
                                       dtype=tf.float32)
            b_logits = tf.get_variable('b_logits', 
                                       shape=[self.config.num_tag], 
                                       dtype=tf.float32, 
                                       initializer=tf.zeros_initializer())
            
            num_step = tf.shape(self.lstm_output)[1]
            word_vec = tf.reshape(self.lstm_output, 
                                  [-1, 2*self.config.word_hidden_size])
            
            # linear
            pred_vec = tf.matmul(word_vec, W_logits) + b_logits

            self.logits = tf.reshape(pred_vec, 
                                     [-1, num_step, self.config.num_tag])
    
    # add pred without crf
    def add_pred_no_crf(self):
        if not self.config.use_crf:
            self.pred_tag = tf.cast(tf.argmax(self.logits, axis=2), tf.int32)
    
    # add loss op
    def add_loss_op(self):
        # with crf
        if self.config.use_crf:
            log_ll, self.transition_param = \
                tf.contrib.crf.crf_log_likelihood(self.logits, self.label, 
                                                  self.seq_len)
            self.loss = tf.reduce_mean(-log_ll)
        
        # mask the loss
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, 
                                                                  self.label)
            mask = tf.sequence_mask(self.seq_len)
            loss_mask = tf.boolean_mask(loss, mask)
            self.loss = tf.reduce_mean(loss_mask)

        # summary
        tf.summary.scalar('loss', self.loss)
    
    # add train op
    def add_train_op(self):
        # optimizer
        with tf.variable_scope('train'):
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
        self.add_word_embed_op()
        self.add_logits_op()
        self.add_pred_no_crf()
        self.add_loss_op()
        self.add_train_op()
        self.add_init_op()
    
    # batch predict
    def predict_batch(self, sess, doc):
        # get the feed dict
        fdict, seq_len = self.build_feed_dict(doc)

        if self.config.use_crf:
            viterbi_seq =[]
            logits, transition_param = sess.run([self.logits, self.transition_param], 
                                                feed_dict=fdict)
            # viterbi loop
            for logit, lseq in zip(logits, seq_len):
                # seq mask
                logit = logit[:lseq]
                viterbi, vscore = tf.contrib.crf.viterbi_decode(logit, 
                                                                transition_param)
                viterbi_seq += [viterbi]
            
            return viterbi_seq, seq_len

        else:
            # prediction without crf
            pred_tag = sess.run(self.pred_tag, feed_dict=fdict)

            return pred_tag, seq_len
    
    # run epoch
    def run_epoch(self, sess, train, dev, entity_dict, epoch):
        # number of batch
        num_batch = (len(train[1]) + self.config.batch_size - 1) // self.config.batch_size

        for idx, (doc, label) in enumerate(create_batch(train, 
                                                        self.config.batch_size)):

            fdict, _ = self.build_feed_dict(doc, label)

            _, train_loss, summary = \
                sess.run([self.train_op, self.loss, self.summary_merge], 
                         feed_dict=fdict)
            
            if idx%10 == 0:
                self.file_writer.add_summary(summary, epoch*num_batch+idx)
                print('epoch {} batch {}, train loss {:04.2f}'.format(epoch, 
                                                                      idx, 
                                                                      train_loss))
            
        accuracy, f1 = self.run_evaluate(sess, dev, entity_dict)
        self.logger.info("dev acc {:04.2f}, f1 {:04.2f}".format(100*accuracy, 
                                                                100*f1))
        return accuracy, f1

    # run evaluate
    def run_evaluate(self, sess, test, entity_dict):

        laccuracy = []
        correct_chunk, all_real, all_pred = 0.0, 0.0, 0.0
        
        for doc, label in create_batch(test, self.config.batch_size):
            label_pred, seq_len = self.predict_batch(sess, doc)

            for y, y_pred, lseq in zip(label, label_pred, seq_len):
                y = y[:lseq]
                y_pred = y_pred[:lseq]

                laccuracy += [r==p for (r, p) in zip(y, y_pred)]
                y_chunk = set(extract_chunk(y, entity_dict))
                y_pred_chunk = set(extract_chunk(y_pred, entity_dict))
                correct_chunk += len(y_chunk & y_pred_chunk)
                all_real += len(y_chunk)
                all_pred += len(y_pred_chunk)

        """
        dev only
        """
        print('correct chunk {}'.format(correct_chunk))
        print('all pred {}'.format(all_pred))
        print('all real {}'.format(all_real))
        
        p = correct_chunk / all_pred if correct_chunk > 0 else 0
        r = correct_chunk / all_real if correct_chunk > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_chunk > 0 else 0

        accuracy = np.mean(laccuracy)
        return accuracy, f1

    # train model
    def train(self, train, dev, entity_dict):

        best_score = 0
        saver = tf.train.Saver()

        # early stop
        num_epoch_no_improve = 0

        with tf.Session() as sess:
            sess.run(self.var_init)

            # load trained model
            if self.config.load_model:
                self.logger.info('load trained model from ' + self.config.model_path)
                saver.restore(sess, self.config.model_path)
            
            # tensorboard
            self.add_summary(sess)

            # train epoch
            for epoch in range(self.config.num_epoch):
                self.logger.info('epoch {} out of {}'.format(epoch+1, 
                                                             self.config.num_epoch))
            
                accuracy, f1 = self.run_epoch(sess, train, dev, entity_dict, epoch)

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

        
    # test model
    def test_model(self, test, entity_dict):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.logger.info("model test")
            saver.restore(sess, self.config.model_path)
            accuracy, f1 = self.run_evaluate(sess, test, entity_dict)
            self.logger.info("test acc {:04.2f}, f1 {:04.2f}".format(100*accuracy, 
                                                                     100*f1))








