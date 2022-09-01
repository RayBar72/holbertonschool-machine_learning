#!/usr/bin/env python3
'''
Moulus that builds, tranis and saves a NN
'''
import numpy as np
import tensorflow.compat.v1 as tf


def shuffle_data(X, Y):
    '''Shuffle data X, Y'''
    m = X.shape[0]
    shuf_vect = np.random.permutation(m)
    x = X[shuf_vect]
    y = Y[shuf_vect]
    return x, y


def create_placeholders(nx, classes):
    '''Function that create a placeholder'''
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y


def create_layer(prev, n, activation):
    '''Function that creates a layer'''
    activa = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=activa, name='layer')
    return layer(prev)


def batch_norm(prev, n, activations, epsilon=1e-8):
    '''Function thar normalizes'''
    activa = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n, kernel_initializer=activa)
    Z = layer(prev)
    mu, sigma_2 = tf.nn.moments(Z, axes=[0])
    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n]),
                        name='gamma')
    beta = tf.Variable(initial_value=tf.constant(0.0, shape=[n]),
                       name='beta')
    Z_b_norm = tf.nn.batch_normalization(
        Z,
        mu,
        sigma_2,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon)
    if activations is None:
        return Z_b_norm
    else:
        return activations(Z_b_norm)


def forward_prop(prev, layers, activations):
    '''Function that makes forward propagation'''
    estimation = batch_norm(prev, layers[0], activations[0])
    for i in range(1, len(layers)):
        if i != len(layers) - 1:
            estimation = batch_norm(estimation, layers[i], activations[i])
        else:
            estimation = create_layer(estimation, layers[i], activations[i])
    return estimation


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    '''Fucntion that calculates Adam'''
    adam = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return adam.minimize(loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''Function that calculates learning rate decay'''
    learning = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                           decay_rate, staircase=True)
    return learning


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
    x, y = create_placeholders(nx, classes)
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection("y_pred", y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection("accuracy", accuracy)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection("loss", loss)
    global_step = tf.Variable(0)
    alpha_d = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha_d, beta1, beta2, epsilon)
    tf.add_to_collection("train_op", train_op)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        m = X_train.shape[0]
        if m % batch_size == 0:
            n_batches = m // batch_size
        else:
            n_batches = m // batch_size + 1
        for i in range(epochs + 1):
            cost_train = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            accuracy_train = sess.run(accuracy,
                                      feed_dict={x: X_train, y: Y_train})
            cost_val = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            accuracy_val = sess.run(accuracy,
                                    feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost_train))
            print("\tTraining Accuracy: {}".format(accuracy_train))
            print("\tValidation Cost: {}".format(cost_val))
            print("\tValidation Accuracy: {}".format(accuracy_val))
            if i < epochs:
                shuffled_X, shuffled_Y = shuffle_data(X_train, Y_train)
                for b in range(n_batches):
                    start = b * batch_size
                    end = (b + 1) * batch_size
                    if end > m:
                        end = m
                    X_mini_batch = shuffled_X[start:end]
                    Y_mini_batch = shuffled_Y[start:end]
                    next_train = {x: X_mini_batch, y: Y_mini_batch}
                    sess.run(train_op, feed_dict=next_train)
                    if (b + 1) % 100 == 0 and b != 0:
                        loss_mini_batch = sess.run(loss, feed_dict=next_train)
                        acc_mini_batch = sess.run(accuracy,
                                                  feed_dict=next_train)
                        print("\tStep {}:".format(b + 1))
                        print("\t\tCost: {}".format(loss_mini_batch))
                        print("\t\tAccuracy: {}".format(acc_mini_batch))
            sess.run(tf.assign(global_step, global_step + 1))
        return saver.save(sess, save_path)
