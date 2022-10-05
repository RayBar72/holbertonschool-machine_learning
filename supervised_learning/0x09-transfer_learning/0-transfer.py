#!/usr/bin/env python3
"""
Transfer Learning.
"""
import tensorflow.keras as K
import numpy as np


def preprocess_data(X, Y):
    """Preprocess data for Resnet50.
    Args:
        X (np.ndarray): matrix of shape (m, 32, 32, 3) containing the CIFAR 10
                        data, where m is the number of data points.
        Y (np.ndarray): matrix of shape (m,) containing the CIFAR 10 labels
                        for X.
    Returns:
        X is a numpy.ndarray containing the preprocessed X
        Y is a numpy.ndarray containing the preprocessed Y
    """
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)

    return X_p, Y_p


if __name__ == '__main__':
    (xtn, ytn), (xtt, ytt) = K.datasets.cifar10.load_data()

    xtn, ytn = preprocess_data(xtn, ytn)
    xtt, ytt = preprocess_data(xtt, ytt)

    model = K.applications.ResNet50(include_top=False,
                                    weights='imagenet',
                                    input_shape=(224, 224, 3))

    for layer in model.layers[:143]:
        layer.trainable = False

    model_1 = K.Sequential()
    model_1.add(K.layers.UpSampling2D((7, 7)))
    model_1.add(model)
    model_1.add(K.layers.AveragePooling2D(pool_size=7))
    model_1.add(K.layers.Flatten())
    model_1.add(K.layers.Dense(10, activation=('softmax')))

    check = K.callbacks.ModelCheckpoint(filepath='cifar10.h5',
                                        monitor='val_acc',
                                        mode='max',
                                        verbose=1,
                                        save_best_only=True)

    model_1.compile(optimizer=K.optimizers.RMSprop(learning_rate=1e-4),
                    loss='categorical_crossentropy',
                    metrics=['acc'])

    model_1.fit(xtn, ytn,
                validation_data=(xtt, ytt),
                batch_size=32,
                epochs=5,
                verbose=1,
                callbacks=[check])

    model_1.save('cifar10.h5')
