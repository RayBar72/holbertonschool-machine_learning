#!/usr/bin/env python3
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


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
