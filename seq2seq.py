# -*- coding:utf-8 -*-
"""
Sample script of Sequence to Sequence model.
You can also use Batch and GPU.
This model is based on below paper.

Ilya Sutskever, Oriol Vinyals, and Quoc V. Le.
Sequence to sequence learning with neural networks.
In Advances in Neural Information Processing Systems (NIPS 2014).
"""
import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable

# global variable (initialize)
xp = np


class Encoder(chainer.Chain):
    # enc=Encoder(vocab_size, feature_num, hidden_num, batch_size),       # encoder
    def __init__(self, vocab_size, embed_size, hidden_size, batch_size):

        print("Encoder called with parameters")
        vocab_size, embed_size, hidden_size, batch_size
        print("vocab size : ", vocab_size)
        print("(embed_size)feature number : ", embed_size)
        print("hidden number : ", hidden_size)
        print("batch size : ", batch_size)
        print()

        super(Encoder, self).__init__(
            # Efficient linear layer for one-hot input.

            # This is a link that wraps the ~chainer.functions.embed_id function. This link holds the ID (word) embedding matrix W as a parameter.

            # Args:
            #     in_size (int): Number of different identifiers (a.k.a. vocabulary
            #         size).
            #     out_size (int): Size of embedding vector.
            #     initialW (initializer <initializer>): Initializer to
            #         initialize the weight. When it is numpy.ndarray, its ndim should be 2.
            #     ignore_label (int or None): If ignore_label is an int value,
            #         i-th row of return value is filled with 0.

            xe=L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            eh=L.Linear(embed_size, 4 * hidden_size),
            hh=L.Linear(hidden_size, 4 * hidden_size),
        )
        self.hidden_size = hidden_size
        self.batch_size = batch_size

    def __call__(self, x, c_pre, h_pre, train=True):
        e = F.tanh(self.xe(x))
        c_tmp, h_tmp = F.lstm(c_pre, self.eh(e) + self.hh(h_pre))
        enable = chainer.Variable(chainer.Variable(x.data != -1).data.reshape(len(x), 1))    # calculate flg whether x is -1 or not
        c_next = F.where(enable, c_tmp, c_pre)                                   # if x!=-1, c_tmp . elseif x=-1, c_pre.
        h_next = F.where(enable, h_tmp, h_pre)                                   # if x!=-1, h_tmp . elseif x=-1, h_pre.
        return c_next, h_next


class Decoder(chainer.Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Decoder, self).__init__(
            ye=L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            eh=L.Linear(embed_size, 4 * hidden_size),
            hh=L.Linear(hidden_size, 4 * hidden_size),
            hf=L.Linear(hidden_size, embed_size),
            fy=L.Linear(embed_size, vocab_size),
        )

    def __call__(self, y, c_pre, h_pre):
        e = F.tanh(self.ye(y))
        c_tmp, h_tmp = F.lstm(c_pre, self.eh(e) + self.hh(h_pre))
        enable = chainer.Variable(chainer.Variable(y.data != -1).data.reshape(len(y), 1))
        c_next = F.where(enable, c_tmp, c_pre)
        h_next = F.where(enable, h_tmp, h_pre)
        f = F.tanh(self.hf(h_next))
        return self.fy(f), c_next, h_next


class Seq2Seq(chainer.Chain):
    def __init__(self, vocab_size, feature_num, hidden_num, batch_size, gpu_flg):
        """
        :param vocab_size: input vocab size
        :param feature_num: size of feature layer (embed layer)
        :param hidden_num: size of hidden layer
        :return:
        """

        print("Seq2seq paased parameters")
        print("vocab size : ", vocab_size)
        print("feature number : ", feature_num)
        print("hidden number : ", hidden_num)
        print("batch size : ", batch_size)
        print()

        global xp
        xp = cuda.cupy if gpu_flg >= 0 else np

        self.vocab_size = vocab_size
        self.hidden_num = hidden_num
        self.batch_size = batch_size
        self.c_batch = Variable(xp.zeros((batch_size, self.hidden_num), dtype=xp.float32))  # cell Variable
        print("cell variable shape : ",self.c_batch.shape)
        self.h_batch = Variable(xp.zeros((batch_size, self.hidden_num), dtype=xp.float32))  # hidden Variable
        print("hidden variable shape : ",self.h_batch.shape)
        print()
        super(Seq2Seq, self).__init__(
            enc=Encoder(vocab_size, feature_num, hidden_num, batch_size),       # encoder
            dec=Decoder(vocab_size, feature_num, hidden_num)                    # decoder
        )
        exit(1)

    def encode(self, input_batch, train):
        """
        Input batch of sequence and update self.c (context vector) and self.h (hidden vector)
        :param input_batch: batch of input text embed id ex.) [[ 1, 0 ,14 ,5 ], [ ...] , ...]
        :param train : True or False
        """
        for batch_word in input_batch:
            batch_word = chainer.Variable(xp.array(batch_word, dtype=xp.int32))
            self.c_batch, self.h_batch = self.enc(batch_word, self.c_batch, self.h_batch, train=train)

    def decode(self, predict_id, teacher_id, train):
        """
        :param predict_id: batch of word ID by output of decoder
        :param teacher_id : batch of correct ID
        :param train: True or false
        :return: decoded embed vector
        """
        batch_word = chainer.Variable(xp.array(predict_id, dtype=xp.int32))
        predict_mat, self.c_batch, self.h_batch = self.dec(batch_word, self.c_batch, self.h_batch)
        if train:
            t = xp.array(teacher_id, dtype=xp.int32)
            t = chainer.Variable(t)
            return F.softmax_cross_entropy(predict_mat, t), predict_mat
        else:
            return predict_mat

    def initialize(self):
        self.c_batch = Variable(xp.zeros((self.batch_size, self.hidden_num), dtype=xp.float32))
        self.h_batch = Variable(xp.zeros((self.batch_size, self.hidden_num), dtype=xp.float32))

    def one_encode(self, src_text, train):
        """
        :param src_text: input text embed id ex.) [ 1, 0 ,14 ,5 ]
        :param train : True or False
        :return: context vector (hidden vector)
        """
        for word in src_text:
            word = chainer.Variable(xp.array([word], dtype=xp.int32))
            self.c_batch, self.h_batch = self.enc(word, self.c_batch, self.h_batch, train=train)

    def one_decode(self, predict_id, teacher_id, train):
        """
        :param predict_id:
        :param teacher_id : embed id ( teacher's )
        :param train: True or false
        :return: decoded embed vector
        """
        word = chainer.Variable(xp.array([predict_id], dtype=xp.int32))
        predict_vec, self.c_batch, self.h_batch = self.dec(word, self.c_batch, self.h_batch)
        if train:
            t = xp.array([teacher_id], dtype=xp.int32)
            t = chainer.Variable(t)
            return F.softmax_cross_entropy(predict_vec, t), predict_vec
        else:
            return predict_vec

    def generate(self, src_text, sentence_limit, word2id, id2word):
        """
        :param src_text: input text embed id ex.) [ 1, 0 ,14 ,5 ]
        :param sentence_limit:
        :param word2id:
        :param id2word:
        :return:
        """
        self.initialize()
        self.one_encode(src_text, train=False)

        sentence = ""
        word_id = word2id["<start>"]
        for _ in range(sentence_limit):
            predict_vec = self.one_decode(predict_id=word_id, teacher_id=None, train=False)
            word = id2word[xp.argmax(predict_vec.data)]     # choose word_ID which has the highest probability
            word_id = word2id[word]
            if word == "<eos>":
                break
            sentence = sentence + word + " "
        return sentence

