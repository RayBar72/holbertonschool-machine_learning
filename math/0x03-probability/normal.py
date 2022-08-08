#!/usr/bin/env python3
'''Module that represents a normal distribution'''


class Normal:
    '''Class that represents a normal distribution'''
    pi = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        '''Class constructor'''
        if data is None:
            self.mean = float(mean)
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            else:
                self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                x = 0
                for i, y in enumerate(data):
                    x += y
                i +=1
                x = x / i
                sigma = 0
                for z in data:
                    sigma += (z -x) ** 2
                sigma = (sigma / i) ** 0.5
                self.mean = float(x)
                self.stddev = float(sigma)

    def z_score(self, x):
        '''Returns the z score of x'''
        score = x - self.mean
        score = score / self.stddev
        return score

    def x_value(self, z):
        '''Returns the x value for a given z score'''
        value = z * self.stddev
        value += self.mean
        return value

    def pdf(self, x):
        '''Calculates the value of the PDF for a given x-value'''
        stand = self.z_score(x)
        const = 1 / (self.stddev * ((2 * Normal.pi) ** 0.5))
        expon = (- 1 / 2) * (stand ** 2)
        y = Normal.e ** expon
        return const * y

    def cdf(self, x):
        '''Claculates the value of the CDF for a given x-value'''
        y = (x - self.mean) / ((self.stddev) * (2 ** 0.5))
        const = 2 / (Normal.pi ** 0.5)
        poli = y - ((y ** 3) / 3) + ((y ** 5) / 10) - ((y ** 7) / 42) + ((y ** 9) / 216)
        return ((1 + const * poli) * (1 / 2))
