#!/usr/bin/env python3
"""
Modulus that creates an encoder block for a transformer
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    Class EncoderBlock
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Returns: a tensor of shape within the encoder block
        """
        atn_out, _ = self.mha(x, x, x, mask)
        atn_out = self.dropout1(atn_out, training=training)
        out1 = self.layernorm1(x + atn_out)
        ffn_out = self.dense_hidden(out1)
        ffn_out = self.dense_output(ffn_out)
        ffn_out = self.dropout2(ffn_out, training=training)
        output = self.layernorm2(out1 + ffn_out)
        return output
