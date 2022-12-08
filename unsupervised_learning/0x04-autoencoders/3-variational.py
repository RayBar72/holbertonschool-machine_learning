#!/usr/bin/env python3
'''
Modulus that creates variational autoencoder
'''
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    '''
    Function that creates variational autoencoder
    '''
    inputs0 = keras.Input(shape=(input_dims,))
    x = inputs0
    for l in hidden_layers:
        x = keras.layers.Dense(l, activation='relu')(x)
    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_sigma = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        '''
        Function that perfors sample of normal
        '''
        z_m, z_stand_des = args
        batch = keras.backend.shape(z_m)[0]
        dim = keras.backend.int_shape(z_m)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_m + keras.backend.exp(z_stand_des / 2) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))([z_mean, z_log_sigma])

    encoder = keras.Model(inputs=inputs0, outputs=[z, z_mean, z_log_sigma])

    inputs1 = keras.Input(shape=(latent_dims,))
    x = inputs1
    for l in reversed(hidden_layers):
        x = keras.layers.Dense(l, activation='relu')(x)
    outputs1 = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(inputs1, outputs1)

    auto = keras.Model(inputs=inputs0, outputs=decoder(encoder(inputs0)[-1]))

    # kl = keras.losses.KLDivergence()

    # auto.compile('adam', kl)

    def lost(x, y):
        '''
        Function that calculates loss function
        '''
        x_bin = keras.backend.binary_crossentropy(x, y)
        x_bin = keras.backend.sum(x_bin, axis=1)
        kl = -0.5 * keras.backend.mean(1 + z_log_sigma -
                                       keras.backend.square(z_mean) -
                                       keras.backend.exp(z_log_sigma),
                                       axis=-1)
        return x_bin + kl

    auto.compile('adam', loss=lost)

    return encoder, decoder, auto
