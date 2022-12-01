#!/usr/bin/env python3
'''
Modulus that perfors bayesian optimization
'''
import numpy as np
from scipy.stats import norm
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

    def acquisition(self):
        '''
        Function that calculates the next best sample location
        '''
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize is True:
            Y_sample = np.min(self.gp.Y)
            im = Y_sample - mu - self.xsi
        else:
            Y_sample = np.max(self.gp.Y)
            im = mu - Y_sample - self.xsi

        ss = sigma.shape[0]
        Z = np.zeros(ss)
        for i in range(ss):
            if sigma[i] > 0:
                Z[i] = im[i] / sigma[i]
            else:
                Z[i] = 0
            expo = im * norm.cdf(Z) + sigma * norm.pdf(Z)

        X_next = self.X_s[np.argmax(expo)]

        return X_next, expo
