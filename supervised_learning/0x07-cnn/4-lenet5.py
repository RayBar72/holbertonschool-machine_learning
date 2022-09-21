#!/usr/bin/env python3
'''
Modulus that builds a modified version of the LeNet-5 architecture
using tensorflow
'''
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    '''
    Function that builds a modified version of the LeNet-5 architecture
    using tensorflow
    '''
    init = tf.keras.initializers.VarianceScaling(scale=2.0)

    C1 = tf.layers.Conv2D(filters=6,
                          kernel_size=5,
                          padding='same',
                          activation=tf.nn.relu,
                          kernel_initializer=init)(x)

    S2 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))(C1)

    C3 = tf.layers.Conv2D(filters=16,
                          kernel_size=5,
                          padding='valid',
                          activation=tf.nn.relu,
                          kernel_initializer=init)(S2)

    S4 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))(C3)

    flatten = tf.layers.Flatten()(S4)

    C5 = tf.layers.Dense(units=120,
                         activation=tf.nn.relu,
                         kernel_initializer=init)(flatten)

    F6 = tf.layers.Dense(units=84,
                         activation=tf.nn.relu,
                         kernel_initializer=init)(C5)

    OUTPUT = tf.layers.Dense(units=10,
                             kernel_initializer=init)(F6)

    OUTPUT_SOFT = tf.nn.softmax(OUTPUT)

    cost = tf.losses.softmax_cross_entropy(y, OUTPUT)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(OUTPUT, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    return OUTPUT_SOFT, optimizer, cost, accuracy