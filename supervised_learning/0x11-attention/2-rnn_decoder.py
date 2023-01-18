#!/usr/bin/env python3
"""
Modulus that creates the class RNNDecoder to decode for machine translation
"""
SelfAttention = __import__('1-self_attention').SelfAttention
import tensorflow as tf


class RNNDecoder(tf.keras.layers.Layer):
    """
    Class that creates the class RNNDecoder to decode for machine translation
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        """
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       return_state=True,
                                       return_sequences=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Function that builds the decoder for machine translation
        """