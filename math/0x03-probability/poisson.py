#!/usr/bin/env python3
'''Module that represents a poisson distribution'''


class Poisson:
    '''Class that represents a Poisson distribution'''
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        '''Class constructor'''
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                x = sum(data) / len(data)
                self.lambtha = x

    def pmf(self, k):
        '''Calculates the value of the PMF for a given number of “successes”'''
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        x = 1
        for i in range(1, k + 1):
            x *= i
        uno = Poisson.e ** (- self.lambtha)
        dos = self.lambtha ** k
        return (uno * dos) / x

    def cdf(self, k):
        '''Calculates the value of the CDF for a given number of “successes”'''
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        lista = []
        for i in range(k + 1):
            lista.append(self.pmf(i))
        return sum(lista)
