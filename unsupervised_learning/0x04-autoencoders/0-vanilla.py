#!/usr/bin/env python3
'''
Modulus that creates a "vanilla" autoencoder
'''
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    '''
    Function that creates a "vanilla" autoencoder
    '''

    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for l in hidden_layers:
        x = keras.layers.Dense(l, activation='relu')(x)
    outputs = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(inputs, outputs)

    inputs2 = keras.Input(shape=(latent_dims,))
    x = inputs2
    for l in reversed(hidden_layers):
        x = keras.layers.Dense(l, activation='relu')(x)
    outputs2 = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs2, outputs2)

    auto = keras.Model(inputs=inputs, outputs=decoder(encoder(inputs)))

    auto.compile(loss='binary_crossentropy',
                 optimizer='adam')

    return encoder, decoder, auto
