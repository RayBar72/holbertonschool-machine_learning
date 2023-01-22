#!/usr/bin/env python3
"""
Modulus that creates the class RNNDecoder to decode for machine translation
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    Class that creates the class RNNDecoder to decode for machine translation
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        """
        super(RNNDecoder, self).__init__()
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
        batch, units = s_prev.shape
        atento = SelfAttention(units)
        context, weights = atento(s_prev, hidden_states)
        embed = self.embedding(x)
        exponencial = tf.expand_dims(context, 1)
        concatenado = tf.concat([exponencial, embed], axis=-1)
        output, state = self.gru(concatenado)
        output = tf.reshape(output, (output.shape[0], output.shape[2]))
        y = self.F(output)
        return y, state
