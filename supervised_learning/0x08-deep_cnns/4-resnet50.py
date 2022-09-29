#!/usr/bin/env python3
'''
Modulus that builds the ResNet-50 architecture as described in Deep
Residual Learning for Image Recognition (2015)
'''
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    '''
    Function that builds the ResNet-50 architecture as described
    in Deep Residual Learning for Image Recognition (2015)

    Returns
    -------
    Keras model.

    '''
    # model = K.applications.resnet50.ResNet50(include_top=True,
    #                                          weights=None,
    #                                          pooling=None,
    #                                          classifier_activation='softmax')
    # return model
    inputs = K.Input(shape=(224, 224, 3))

    A = K.layers.Conv2D(64,
                        kernel_size=(7, 7),
                        strides=2,
                        padding='same',
                        kernel_initializer='he_normal')(inputs)
    A = K.layers.BatchNormalization(axis=3)(A)
    A = K.layers.Activation('relu')(A)
    A = K.layers.MaxPool2D(pool_size=(3, 3),
                           strides=2,
                           padding='same')(A)

    A = projection_block(A, [64, 64, 256], s=1)
    A = identity_block(A, [64, 64, 256])
    A = identity_block(A, [64, 64, 256])

    A = projection_block(A, [128, 128, 512], s=2)
    A = identity_block(A, [128, 128, 512])
    A = identity_block(A, [128, 128, 512])
    A = identity_block(A, [128, 128, 512])

    A = projection_block(A, [256, 256, 1024], s=2)
    A = identity_block(A, [256, 256, 1024])
    A = identity_block(A, [256, 256, 1024])
    A = identity_block(A, [256, 256, 1024])
    A = identity_block(A, [256, 256, 1024])
    A = identity_block(A, [256, 256, 1024])

    A = projection_block(A, [512, 512, 2048], s=2)
    A = identity_block(A, [512, 512, 2048])
    A = identity_block(A, [512, 512, 2048])

    A = K.layers.AveragePooling2D((7, 7), strides=1)(A)
    A = K.layers.Dense(1000,
                       activation='softmax',
                       kernel_initializer='he_normal')(A)

    model = K.Model(inputs, A)

    return model
