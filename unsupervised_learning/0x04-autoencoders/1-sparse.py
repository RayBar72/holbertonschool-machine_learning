#!/usr/bin/env python3
'''
Modulus that creates a sparse
'''
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    '''
    Function that creates a sparse autoencoder
    '''
    l1 = keras.regularizers.l1(lambtha)
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for l in hidden_layers:
        x = keras.layers.Dense(l, 'relu')(x)
    outputs = keras.layers.Dense(latent_dims, 'relu', kernel_regularizer=l1)(x)
    encoder = keras.Model(inputs, outputs)

    inputs2 = keras.Input(shape=(latent_dims,))
    x = inputs2
    for l in reversed(hidden_layers):
        x = keras.layers.Dense(l, 'relu')(x)
    outputs2 = keras.layers.Dense(input_dims,
                                  'sigmoid')(x)
    decoder = keras.Model(inputs2, outputs2)

    auto = keras.Model(inputs=inputs, outputs=decoder(encoder(inputs)))

    auto.compile(loss='binary_crossentropy',
                 optimizer='adam')

    return encoder, decoder, auto
