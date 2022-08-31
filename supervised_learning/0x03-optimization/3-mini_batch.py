#!/usr/bin/env python3
'''Function that trains a loaded neural network'''
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    '''
    Train a mini-batch of training data
    Args:
        X_train: numpy.ndarray of shape (m, 784)
                 contains training data
        Y_train: one-hot numpy.ndarray of shape (m, 10)
                 contains training labels
        X_valid: numpy.ndarray of shape (m, 784)
                 contains validation data
        Y_valid: one-hot numpy.ndarray of shape (m, 10)
                 contains validation labels
        batch_size: type int number of data points in a batch
        epochs: type int number of times the training should pass
                through the whole dataset
        load_path: path from which to load the model
        save_path: path to where the model should be saved after training
    Returns: path where the model was saved
    '''
    with tf.Session() as session:
        saved = tf.train.import_meta_graph('{}.meta'.format(load_path))
        saved.restore(session, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        m = X_train.shape[0]
        if m % batch_size == 0:
            mini = int(m / batch_size)
            case = 0
        else:
            mini = int(m / batch_size) + 1
            case = 1
        for e in range(epochs + 1):
            d_train = {x: X_train, y: Y_train}
            d_valid = {x: X_valid, y: Y_valid}
            t_cost = session.run(loss, feed_dict=d_train)
            t_accuracy = session.run(accuracy, feed_dict=d_train)
            t_train_op = session.run(loss, feed_dict=d_train)
            v_cost = session.run(loss, feed_dict=d_valid)
            v_accuracy = session.run(accuracy, feed_dict=d_valid)
            print('After {} epochs:'.format(e))
            print('\tTraining Cost: {}'.format(t_cost))
            print('\tTraining Accuracy: {}'.format(t_accuracy))
            print('\tValidation Cost: {}'.format(v_cost))
            print('\tValidation Accuracy: {}'.format(v_accuracy))
            if e < epochs:
                Xsh, Ysh = shuffle_data(X_train, Y_train)
                for i in range(mini):
                    a_0 = i * batch_size
                    a_1 = (i + 1) * batch_size
                    if case == 0 and i == mini:
                        x_m = Xsh[a_0::]
                        y_m = Ysh[a_0::]
                    else:
                        x_m = Xsh[a_0:a_1]
                        y_m = Ysh[a_0:a_1]
                    d_mini = {x: x_m, y: y_m}
                    session.run(train_op, feed_dict=d_mini)
                    if((i + 1) % 100 == 0) and (i != 0):
                        s_cost = session.run(loss, feed_dict=d_mini)
                        s_accuracy = session.run(accuracy, feed_dict=d_mini)
                        print('\tStep {}:'.format(i + 1))
                        print('\t\tCost: {}'.format(s_cost))
                        print('\t\tAccuracy: {}'.format(s_accuracy))
        save_path = saved.save(session, save_path)
    return save_path
