#!/usr/bin/env python3
'''
Function that plots a line graph:
    The x-axis should be labeled Time (years)
    The y-axis should be labeled Fraction Remaining
    The title should be Exponential Decay of C-14
    The y-axis should be logarithmically scaled
    The x-axis should range from 0 to 28650
'''
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

plt.plot(x, y, '-')
plt.title('Exponential Decay of C-14')
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.yscale('log')
plt.xlim([0, 28650])
