#!/usr/bin/env python3
'''
Modulus that creates a class Yolo
'''
import tensorflow.keras as K


class Yolo():
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            class_names = f.read().split('\n')
        self.class_names = class_names
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
