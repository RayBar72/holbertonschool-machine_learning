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
    layer = tf.layers.Dense(n, activation=None,
                            kernel_initializer=activa, name='layer')
    return layer(prev)

def batch_norm(prev, n, activations, epsilon):
    '''Function thar normalizes'''
    activa = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(n, activation=None,
                            kernel_initializer=activa, name='layer')
    Z = layer(prev)
    mu, sigma_2 = tf.nn.moments(Z, axes=[0])
    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n]),
                        name='gamma')
    beta = tf.Variable(initial_value=tf.constant(0.0, shape=[n]),
                       name='beta')
    Z_b_norm = tf.nn.batch_normalization(
        Z, mu,
        sigma_2,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon)
    return activations(Z_b_norm)
    
def forward_prop(prev, layers, activations, epsilon):
    '''Function that makes forward propagation'''
    estimation = batch_norm(prev, layers[0], activations[0], epsilon)
    for i in range(1, len(layers)):
        if i == len(layers) - 1:
            estimation = create_layer(estimation, layers[i], activations[i])
        else:
            estimation = batch_norm(estimation, layers[i],
                                    activations[i], epsilon)
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

    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    x, y = create_placeholders(nx, classes)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layers, activations, epsilon)
    tf.add_to_collection('y_pred', y_pred)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    global_step = tf.Variable(0)
    alpha_dec = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha_dec, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init)
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
            session.run(tf.assign(global_step, global_step + 1))
            save_path = saver.save(session, save_path)
    return save_path
