#!/usr/bin/env python3
'''Modulus that represents a binomial distribution'''


class Binomial:
    '''Class that represents the binomial distribution'''
    # pi = 3.1415926536
    def __init__(self, data=None, n=1, p=0.5):
        '''Class constructor'''
        if data is None:
            n = int(n)
            if n < 1:  # Ojo está con valor que incluye cero
                raise ValueError('n must be a positive value')
            if p >= 1 or p <= 0:
                raise ValueError('p must be greater than 0 and less than 1')
            else:
                self.p = float(p)
                self.n = int(n)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            # Calculating media
            x = 0
            for i in data:
                x += i
            x = x / len(data)
            # Calculating var
            sigmak = 0
            for j in data:
                sigmak += (x - j) ** 2
            sigmak = sigmak / len(data)
            # Calculating p
            p = 0
            p = 1 - (sigmak / x)
            # Calculating n
            n = 0
            n = x / p
            n = int(round(n, 0))
            self.n = n
            self.p = x / n

    def factorial(x):
        '''Calculates factorial of a given number'''
        y = 1
        for i in range(1, x + 1):
            y *= i
        return y

    def pmf(self, k):
        '''Calculates the value of the PMF for a given number of “successes”'''
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        div = self.factorial(k) * self.factorial(self.n - k)
        coci = self.factorial(self.n)
        mult = (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return (coci / div) * mult

    def cdf(self, k):
        '''Calculates the value of the CDF for a given number of “successes”'''
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        x = 0
        for i in range(0, k + 1):
            x += self.pmf(i)
        return x
