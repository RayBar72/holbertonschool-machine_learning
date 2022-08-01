#!/usr/bin/env python3
'''Function that graphs a stacked bar graph'''
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

frname = ['apples', 'bananas', 'oranges', 'peaches']
people = ['Farrah', 'Fred', 'Felicia']
colores = ['r', 'y', '#ff8000', '#ffe5b4']

for i in range(len(fruit)):
    plt.bar(people,
            fruit[i],
            bottom=np.sum(fruit[:i], axis=0),
            color=colores[i],
            label=frname[i],
            width=0.5)

plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')
plt.yticks(range(0, 81, 10))
plt.legend()
plt.show()
