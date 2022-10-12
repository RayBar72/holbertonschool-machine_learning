#!/usr/bin/env python3
'''
Modulus that creates a class Yolo
'''
import tensorflow.keras as K


class Yolo():
    '''
    Class that uses YOLO v3 algorithm to perform object detection
    '''
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        '''
        Function thas initializes Yolo class
            model_path - where Darkenet Keras model is stored
        classes_path - where list of clasess is names are stored
        class_t - float with box score threshold
        nms_t - float with IOU threshold for non-max suppresion
        anchor - np.array of shape (outputs, anchor_boxes, 2)
            outputs - number of predictions made by Darknet model
            anchor_boxes - number of anchor bosx used for each prediction
            2 - [anchor_box_width, anchor_box_height]
        public instances:
            model
            class_names
            class_t
            nms_t
            anchors
        '''
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            class_names = f.read().split('\n')
        self.class_names = class_names
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
