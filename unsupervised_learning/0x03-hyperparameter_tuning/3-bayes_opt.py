#!/usr/bin/env python3
'''
Modulus that perfors bayesian optimization
'''
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    '''
    Class that performs bayesian optimization
    '''
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        '''
        Function init for bayesian optimization
        '''
        self.f = f

        self.gp = GP(X_init, Y_init, l, sigma_f)

        X_s = np.linspace(bounds[0], bounds[1], num=ac_samples)
        self.X_s = X_s.reshape(-1, 1)

        self.xsi = xsi

        self.minimize = minimize
