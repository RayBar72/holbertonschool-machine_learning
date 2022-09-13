#!/usr/bin/env python3
'''
Modulus that trins a model using mini-batch gradien descent
'''
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    '''
    Based on 7-train.py, update the function train_model
    to also analyze validaiton data
    '''
    def decay(epoch):
        '''
        Function that calculates step decay
        '''
        return alpha / (1 + decay_rate * epoch)
    callb = []
    if validation_data and learning_rate_decay:
        lr = K.callbacks.LearningRateScheduler(decay, verbose=1)
        callb.append(lr)
    if validation_data and early_stopping:
        validation_data = validation_data
        early = K.callbacks.EarlyStopping(monitor='val_loss',
                                          patience=patience)
        callb.append(early)
    if save_best:
        best = K.callbacks.ModelCheckpoint(filepath, save_best_only=True)
        callb.append(best)
    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       validation_data=validation_data,
                       shuffle=shuffle,
                       callbacks=callb)
