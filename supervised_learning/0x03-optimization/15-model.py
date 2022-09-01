#!/usr/bin/env python3
'''
Moulus that builds, tranis and saves a NN
'''
import numpy as np
import tensorflow.compat.v1 as tf


def shuffle_data(X, Y):
    '''Shuffle data X, Y'''
    m = X.shape[0]
    shuf_vect = list(np.random.permutation(m))
    x = X[shuf_vect, :]
    y = Y[shuf_vect, :]
    return x, y


def create_layer(prev, n, activation):
    '''Function that creates a layer'''
    activa = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=activa, name='layer')
    return layer(prev)


def create_batch_norm_layer(prev, n, activation):
    '''Function thar normalizes'''
    activa = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n, kernel_initializer=activa)
    Z = layer(prev)
    mu, sigma_2 = tf.nn.moments(Z, axes=[0])
    epsilon = 1e-8
    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n]),
                        name='gamma', trainable=True)
    beta = tf.Variable(initial_value=tf.constant(0.0, shape=[n]),
                       name='beta', trainable=True)
    Z_b_norm = tf.nn.batch_normalization(
        Z,
        mu,
        sigma_2,
        beta,
        gamma,
        epsilon)
    if activation is None:
        return Z_b_norm
    return activation(Z_b_norm)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''Function that calculates learning rate decay'''
    learning = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                           decay_rate, staircase=True)
    return learning


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    '''Fucntion that calculates Adam'''
    adam = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return adam.minimize(loss)


def forward_prop(prev, layers=[], activations=[]):
    '''Function that makes forward propagation'''
    estimation = create_batch_norm_layer(prev, layers[0], activations[0])
    for i in range(1, len(layers)):
        if i != len(layers) - 1:
            estimation = create_batch_norm_layer(estimation, layers[i],
                                                 activations[i])
        else:
            estimation = create_layer(estimation, layers[i], activations[i])
    return estimation


def calculate_accuracy(y, y_pred):
    '''Function that calculates accuracy'''
    yes_not = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    acura = tf.reduce_mean(tf.cast(yes_not, tf.float32))
    return acura


def calculate_loss(y, y_pred):
    '''Fuction that calculates loss'''
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    '''Function that trains NN'''
    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]
    (X_train, Y_train) = Data_train
    (X_valid, Y_valid) = Data_valid
    x = tf.placeholder(tf.float32, shape=[None, nx], name="x")
    tf.add_to_collection("x", x)
    y = tf.placeholder(tf.float32, shape=[None, classes], name="y")
    tf.add_to_collection("y", y)
    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection("y_pred", y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection("accuracy", accuracy)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection("loss", loss)
    global_step = tf.Variable(0, trainable=False)
    alpha_d = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha_d, beta1, beta2, epsilon)
    tf.add_to_collection("train_op", train_op)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    m = X_train.shape[0]
    if m % batch_size == 0:
        complete = 1
        num_batches = int(m / batch_size)
    else:
        complete = 0
        num_batches = int(m / batch_size) + 1
    with tf.Session() as sess:
        sess.run(init)
        for i in range(epochs + 1):
            feed_t = {x: X_train, y: Y_train}
            feed_v = {x: X_valid, y: Y_valid}
            train_cost, train_accuracy = sess.run([loss, accuracy], feed_t)
            valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_v)
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))
            if i < epochs:
                X_shu, Y_shu = shuffle_data(X_train, Y_train)
                for k in range(num_batches):
                    if complete == 0 and k == num_batches - 1:
                        start = k * batch_size
                        X_minibatch = X_shu[start::]
                        Y_minibatch = Y_shu[start::]
                    else:
                        start = k * batch_size
                        end = (k * batch_size) + batch_size
                        X_minibatch = X_shu[start:end]
                        Y_minibatch = Y_shu[start:end]
                    feed_mb = {x: X_minibatch, y: Y_minibatch}
                    sess.run(train_op, feed_mb)
                    if (k + 1) % 100 == 0 and k != 0:
                        mb_c, mb_a = sess.run([loss, accuracy], feed_mb)
                        print("\tStep {}:".format(k + 1))
                        print("\t\tCost: {}".format(mb_c))
                        print("\t\tAccuracy: {}".format(mb_a))
            sess.run(tf.assign(global_step, global_step + 1))
        return saver.save(sess, save_path)
