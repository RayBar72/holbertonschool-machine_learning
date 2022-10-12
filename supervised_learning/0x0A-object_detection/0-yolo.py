#!/usr/bin/env python3
'''
Modulus that creates a class Yolo
'''
import tensorflow.keras as K


class Yolo():
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = 