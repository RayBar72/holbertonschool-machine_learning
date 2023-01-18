#!/usr/bin/env python3
"""
Modulus that enconde for machine translation
"""
import numpy as np
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    class that inherits from tensorflow.keras.layers.Layer to encode for machine translation
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        class constructor
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """
        Public instance method that initializes the hidden states for the RNN cell to a tensor of zeros
        """
        initial = tf.keras.initializers.Zeros()
        return initial(shape=(self.batch, self.units))

    def call(self, x, initial):
        """
        Public instance method that returns outputs, hidden
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
