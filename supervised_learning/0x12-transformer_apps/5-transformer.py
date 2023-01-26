#!/usr/bin/env python3
"""
Modulus for the class Transformer
"""
import tensorflow as tf
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Function that calculates the positional encoding for a transformer
    """
    t = np.arange(max_seq_len).reshape(max_seq_len, 1)
    indice = np.arange(dm).reshape(1, dm)
    dm_float = np.float32(dm)

    W = 1 / (np.power(10000, (2*(indice//2)/dm_float)))

    Wt = (W * t)

    positional = np.zeros((max_seq_len, dm))

    positional[:, 0::2] = np.sin(Wt[:, 0::2])

    positional[:, 1::2] = np.cos(Wt[:, 1::2])

    return positional


def sdp_attention(Q, K, V, mask=None):
    """
    Function that calculates the scaled dot product attention
    """
    uno = tf.matmul(Q, K, transpose_b=True)
    k = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled = uno / tf.math.sqrt(k)

    if mask is not None:
        scaled += (mask * -1e9)

    weights = tf.nn.softmax(scaled, axis=-1)

    output = tf.matmul(weights, V)

    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Class MultiHeadAttention
    """
    def __init__(self, dm, h):
        """
        Class constructor
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Returns: output, weights
        """
        batch_size = tf.shape(Q)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        output, weights = sdp_attention(Q, K, V, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output,
                                      (batch_size, -1, self.dm))
        output = self.linear(concat_attention)
        return output, weights


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


class Encoder(tf.keras.layers.Layer):
    """
    Class to create the encoder for a transformer
    """
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        Class constructor
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden,
                                    drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Function to create the encoder for a transformer
        """
        len_seq = x.shape[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:len_seq]
        x = self.dropout(x, training=training)
        for i in range(self.N):
            x = self.blocks[i](x, training, mask)
        return x


class Decoder(tf.keras.layers.Layer):
    """
    Class to create the decoder for a transformer
    """
    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        Class constructor
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden,
                                    drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Function to create the decoder for a transformer
        """
        len_seq = x.shape[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:len_seq]
        x = self.dropout(x, training=training)
        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training,
                               look_ahead_mask, padding_mask)
        return x


class Transformer(tf.keras.Model):
    """
    Class to create a transformer network
    """
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Class constructor
        """
        super(Transformer, self).__init__()
        self.N = N
        self.dm = dm
        self.h = h
        self.hidden = hidden
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab
        self.max_seq_input = max_seq_input
        self.max_seq_target = max_seq_target
        self.drop_rate = drop_rate

        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)

        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)

        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """
        Function to create a transformer network
        """
        enc_out = self.encoder(inputs, training, encoder_mask)
        dec_out = self.decoder(target, enc_out, training, look_ahead_mask,
                               decoder_mask)
        output = self.linear(dec_out)
        return output
