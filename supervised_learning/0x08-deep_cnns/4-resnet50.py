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
    model = K.applications.resnet50.ResNet50(include_top=True,
                                             weights=None,
                                             pooling=None,
                                             classifier_activation='softmax')
    return model
