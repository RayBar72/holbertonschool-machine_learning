#!/usr/bin/env python3
'''
Class that defines a  neural network with one
hidden layer performing binary classification
'''
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork():
    '''Class that defines a NN whit one hieden layer'''
    def __init__(self, nx, nodes):
        '''Class constructor'''
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros([nodes, 1])
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        '''W1 getter function'''
        return self.__W1

    @property
    def b1(self):
        '''b1 getter function'''
        return self.__b1

    @property
    def A1(self):
        '''A1 getter function'''
        return self.__A1

    @property
    def W2(self):
        '''W2 getter function'''
        return self.__W2

    @property
    def b2(self):
        '''b2 getter function'''
        return self.__b2

    @property
    def A2(self):
        '''A1 getter function'''
        return self.__A2

    def forward_prop(self, X):
        '''
        Calculates the forward propagation of the neural network
        '''
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        '''
        Calculates the cost of the model using logistic regression
        '''
        m = Y.shape[1]
        C = - (1 / m) * (np.sum(Y * np.log(A) + (1 - Y) *
                                (np.log(1.0000001 - A))))
        return C

    def evaluate(self, X, Y):
        '''
        Evaluates the neural network’s predictions
        '''
        self.forward_prop(X)
        P = np.where(self.__A2 >= 0.5, 1, 0)
        return P, self.cost(Y, self.__A2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        '''
        Calculates one pass of gradient descent on the neural network
        '''
        m = Y.shape[1]
        dz2 = A2 - Y
        dW2 = (1 / m) * np.matmul(A1, dz2.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dW1 = (1 / m) * np.matmul(dz1, X.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
        self.__W2 = self.__W2 - (alpha * dW2).T
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        '''
        Trains neural network
        '''
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step < 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        iteration = []
        c = []
        for i in range(iterations + 1):
            a, cost = self.evaluate(X, Y)
            self.forward_prop(X)
            if i % step == 0:
                iteration.append(i)
                c.append(cost)
                if verbose:
                    print('Cost after {} iterations: {}'.format(i, cost))
            if i < iterations:
                self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
        if graph:
            plt.plot(iteration, c, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
