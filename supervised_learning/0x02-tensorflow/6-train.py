#!/usr/bin/env python3
'''
Modulus that builds, trains, and saves a neural network classifier
'''
import tensorflow.compat.v1 as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    '''
    Function that that builds, trains, and saves a neural network classifier

    Parameters
    ----------
    X_train : numpy.ndarray
        Input data.
    Y_train : numpy.ndarray
        Training labels.
    X_valid : numpy.ndarray
        Validation input data.
    Y_valid : numpy.ndarray
        Validation labels.
    layer_sizes : List
        Numbers of nodes in each layer.
    activations : List
        Activations functions.
    alpha : Float
        Learning rate.
    iterations : Int
        Number of iterations in the training.
    save_path : Str
        Path to save the model.

    Returns
    -------
    Path where the model was saved
    '''
    # Creating tensor for inputs, graph and funct
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)
    # Initializing, saver, session and running
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    session = tf.Session()
    session.run(init)
    for i in range(iterations + 1):
        tr_loss = session.run(loss, feed_dict={x: X_train, y: Y_train})
        tr_act = session.run(accuracy, feed_dict={x: X_train, y: Y_train})
        vl_loss = session.run(loss, feed_dict={x: X_valid, y: Y_valid})
        vl_act = session.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
        if i % 100 == 0 or i == iterations:
            print('After {} iterations:'.format(i))
            print('\tTraining Cost: {}'.format(tr_loss))
            print('\tTraining Accuracy: {}'.format(tr_act))
            print('\tValidation Cost: {}'.format(vl_loss))
            print('\tValidation Accuracy: {}'.format(vl_act))
        if i < iterations:
            session.run(train_op, feed_dict={x: X_train, y: Y_train})
    return saver.save(session, save_path)
