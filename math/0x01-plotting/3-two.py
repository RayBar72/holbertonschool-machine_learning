#!/usr/bin/env python3
'''
Function that plots x ↦ y1 and x ↦ y2 as line graphs:
    The x-axis should be labeled Time (years)
    The y-axis should be labeled Fraction Remaining
    The title should be Exponential Decay of Radioactive Elements
    The x-axis should range from 0 to 20,000
    The y-axis should range from 0 to 1
    x ↦ y1 should be plotted with a dashed red line
    x ↦ y2 should be plotted with a solid green line
    A legend labeling x ↦ y1 as C-14 and x ↦ y2 as Ra-226 should
    be placed in the upper right hand corner of the plot
'''
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

fig, ax = plt.subplots()
uno, = plt.plot(x, y1, '--r', label='C-14')
dos, = plt.plot(x, y2, '-g', label='Ra-226')
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of Radioactive Elements')
plt.xlim([0, 20000])
plt.ylim([0, 1])
# first_legend = ax.legend(handles=[uno], loc='upper right')
# ax.add_artist(first_legend)
plt.legend(handles=[uno, dos], loc='upper right')
plt.show()
