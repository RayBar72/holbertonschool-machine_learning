#!/usr/bin/env python3
'''
Modulus that builds a DenseNet-121 architecture as described in
"Densenly Connected Convolutional Networks"
'''
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    '''
    Function that builds a DenseNet-121

    Parameters
    ----------
    growth_rate : TYPE, optional
        DESCRIPTION. The default is 32.
    compression : TYPE, optional
        DESCRIPTION. The default is 1.0.

    Returns
    -------
    Keras model.

    '''
    inputs = K.Input(shape=(224, 224, 3))
    nb_filters = 2 * growth_rate
    X = K.layers.BatchNormalization(axis=3)(inputs)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(filters=nb_filters,
                        kernel_size=7,
                        padding='same',
                        strides=2,
                        kernel_initializer='he_normal')(X)
    X = K.layers.AveragePooling2D(pool_size=(3, 3),
                                  strides=2,
                                  padding='same')(X)

    dense_b = [6, 12, 24, 16]
    for i in range(4):
        X, nb_filters = dense_block(X,
                                    nb_filters,
                                    growth_rate,
                                    dense_b[i])
        if i != 3:
            X, nb_filters = transition_layer(X,
                                             nb_filters,
                                             compression)

    X = K.layers.AveragePooling2D(pool_size=(7, 7),
                                  padding='same')(X)

    X = K.layers.Dense(units=1000,
                       activation='softmax',
                       kernel_initializer='he_normal')(X)

    model = K.Model(inputs, X)
    return model
