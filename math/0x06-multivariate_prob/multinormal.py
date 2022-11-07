import numpy as np


class MultiNormal():
    def __init__(self, data):
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')

        if data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')

        d, n = data.shape

        self.mean = np.mean(data, axis=1).reshape(d, 1)

        X_m = data - self.mean

        self.cov = np.matmul(X_m, X_m.T) / (n - 1)
