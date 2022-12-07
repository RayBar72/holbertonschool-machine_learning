#!/usr/bin/env python3
'''
Modulus that creates a convolutional autoencoder
'''
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    '''
    Function that creates a convolutional autoencoder
    '''
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for l in filters:
        x = keras.layers.Conv2D(l,
                                (3, 3),
                                padding='same',
                                activation='relu')(x)
        x = keras.layers.MaxPooling2D(l,
                                      (2, 2),
                                      padding='same')(x)
    outputs = x
    encoder = keras.Model(inputs, outputs)

    inputs1 = keras.Input(shape=latent_dims)
    x = inputs1
    reversa = filters[::]
    reversa.reverse()
    for i in range(len(reversa) - 1):
        x = keras.layers.Conv2D(reversa[i],
                                (3, 3),
                                padding='same',
                                activation='relu')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(filters[0],
                            (3, 3),
                            activation='relu')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    outputs1 = keras.layers.Conv2D(input_dims[-1],
                                   (3, 3),
                                   padding='same',
                                   activation='sigmoid')(x)
    decoder = keras.Model(inputs1, outputs1)

    auto = keras.Model(inputs=inputs, outputs=decoder(encoder(inputs)))

    auto.compile(loss='binary_crossentropy',
                 optimizer='adam')

    return encoder, decoder, auto
