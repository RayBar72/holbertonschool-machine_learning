#!/usr/bin/env python3
'''
Modulus that builds a modified version of the LeNet-5
architecture using keras
'''
import tensorflow.keras as K


def lenet5(X):
    '''
    Function that builds a modified version of the LeNet-5
    architecture using keras
    '''
    L = K.layers.Conv2D(6, (5, 5), padding='same', activation='relu',
                        kernel_initializer='he_normal')(X)
    L = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(L)
    L = K.layers.Conv2D(16, (5, 5), padding='valid', activation='relu',
                        kernel_initializer='he_normal')(L)
    L = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(L)
    L = K.layers. Flatten()(L)

    L = K.layers.Dense(120, activation='relu',
                       kernel_initializer='he_normal')(L)
    L = K.layers.Dense(84, activation='relu',
                       kernel_initializer='he_normal')(L)
    L = K.layers.Dense(10, activation='softmax',
                       kernel_initializer='he_normal')(L)

    model = K.Model(X, L)
    adam = K.optimizers.Adam()
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
