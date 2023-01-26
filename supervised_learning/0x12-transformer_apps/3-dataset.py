#!/usr/bin/env python3
"""3-dataset.py"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    Dataset class that loads and preps a dataset for machine translation
    """
    def __init__(self, batch_size, max_len):
        """Init method"""
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.\
            tokenize_dataset(self.data_train)
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        self.data_train = self.data_train.filter(lambda x, y:
                                                 tf.size(x) <= max_len and
                                                 tf.size(y) <= max_len)
        self.data_train = self.data_train.cache()
        largo = len(list(self.data_train.as_numpy_iterator()))
        # print(largo)
        self.data_train = self.data_train.shuffle(largo)
        self.data_train = self.data_train.padded_batch(batch_size,
                                                       ([None], [None]))
        self.data_train = self.data_train.\
            prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        self.data_valid = self.data_valid.filter(lambda x, y:
                                                 tf.size(x) <= max_len and
                                                 tf.size(y) <= max_len)
        self.data_valid = self.data_valid.padded_batch(batch_size,
                                                       ([None], [None]))

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset
        """
        tokenizer_pt = tfds.deprecated.text.\
            SubwordTextEncoder.build_from_corpus(
                (pt.numpy() for pt, en in data), target_vocab_size=2**15)
        tokenizer_en = tfds.deprecated.text.\
            SubwordTextEncoder.build_from_corpus(
                (en.numpy() for pt, en in data), target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes translation into tokens
        """
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Tensorflow wrapper for encoding
        """
        tf_pt, tf_en = tf.py_function(self.encode,
                                      [pt, en],
                                      [tf.int64, tf.int64])
        tf_pt.set_shape([None])
        tf_en.set_shape([None])
        return tf_pt, tf_en
