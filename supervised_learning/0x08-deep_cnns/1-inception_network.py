#!/usr/bin/env python3
'''
Modulus that builds the inception network as described in Going Deeper with
Convolutions (2014)
'''
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    '''
    Function that that builds the inception network as described in
    Going Deeper with Convolutions (2014):

    Returns
    -------
    Keras model.

    '''
    inputs = K.Input(shape=(224, 224, 3))
    C1 = K.layers.Conv2D(64,
                         kernel_size=(7, 7),
                         strides=2,
                         padding='same',
                         kernel_initializer='he_normal',
                         activation='relu')(inputs)
    MP1 = K.layers.MaxPool2D(pool_size=(3, 3),
                             strides=2,
                             padding='same')(C1)
    # C2 = K.layers.Conv2D(64,
    #                      kernel_size=(1, 1),
    #                      strides=1,
    #                      padding='same',
    #                      activation='relu')(MP1)
    C3 = K.layers.Conv2D(192,
                         kernel_size=(3, 3),
                         strides=1,
                         padding='same',
                         kernel_initializer='he_normal',
                         activation='relu')(MP1)
    MP2 = K.layers.MaxPool2D(pool_size=(3, 3),
                             strides=2,
                             padding='same')(C3)
    IN1A = inception_block(MP2,
                           [64, 96, 128, 16, 32, 32])
    IN1B = inception_block(IN1A,
                           [128, 128, 192, 32, 96, 64])
    MP3 = K.layers.MaxPool2D(pool_size=(3, 3),
                             strides=2,
                             padding='same')(IN1B)
    IN2A = inception_block(MP3,
                           [192, 96, 208, 16, 48, 64])
    IN2B = inception_block(IN2A,
                           [160, 112, 224, 24, 64, 64])
    IN2C = inception_block(IN2B,
                           [128, 128, 256, 24, 64, 64])
    IN2D = inception_block(IN2C,
                           [112, 144, 288, 32, 64, 64])
    IN2E = inception_block(IN2D,
                           [256, 160, 320, 32, 128, 128])
    MP4 = K.layers.MaxPool2D(pool_size=(3, 3),
                             strides=2,
                             padding='same')(IN2E)
    IN3A = inception_block(MP4,
                           [256, 160, 320, 32, 128, 128])
    IN3B = inception_block(IN3A,
                           [384, 192, 384, 48, 128, 128])
    AP1 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                    strides=1)(IN3B)
    D1 = K.layers.Dropout(0.4)(AP1)
    SF1 = K.layers.Dense(1000,
                         kernel_initializer='he_normal',
                         activation='softmax')(D1)
    model = K.Model(inputs=inputs, outputs=SF1)
    return model
