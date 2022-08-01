#!/usr/bin/env python3
'''
File that plot y as a line graph:
    y should be plotted as a solid red line
    The x-axis should range from 0 to 10
'''
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

plt.plot(y, color='red')
plt.ylim(-1, 1100)
plt.xlim(0, 10)
plt.show()
