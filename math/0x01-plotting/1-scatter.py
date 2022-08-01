#!/usr/bin/env python3
'''Function that graphs a scatter plot:
    The x-axis should be labeled Height (in)
    The y-axis should be labeled Weight (lbs)
    The title should be Men's Height vs Weight
    The data should be plotted as magenta points
'''
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

plt.plot(x, y, '.m')
plt.title("Men's Height vs Weight")
plt.xlabel("Height (in)")
plt.ylabel("Weight (in)")
plt.show()
