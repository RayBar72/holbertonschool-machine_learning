#!/usr/bin/env python3
"""
Modulus for the class DecoderBlock
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    Class to create a dencoder block for a transformer
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """
        Returns tensor containing the block output
        """
        atn1, _ = self.mha1(x, x, x, look_ahead_mask)
        atn1 = self.dropout1(atn1, training=training)
        out1 = self.layernorm1(x + atn1)
        atn2, _ = self.mha2(out1, encoder_output, encoder_output,
                            padding_mask)
        atn2 = self.dropout2(atn2, training=training)
        out2 = self.layernorm2(out1 + atn2)
        ffn_out = self.dense_hidden(out2)
        ffn_out = self.dense_output(ffn_out)
        ffn_out = self.dropout3(ffn_out, training=training)
        out3 = self.layernorm3(out2 + ffn_out)
        return out3
