#!/usr/bin/env python3
'''Function that graphs a stacked bar graph'''
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

frname = ['apples', 'bananas', 'oranges', 'peaches']
people = ['Farrah', 'Fred', 'Felicia']
colores = ['r', 'y', '#ff8000', '#ffe5b4']

print(fruit)
for i in range(len(frname)):
    for j in range(0, len(people)):
        plt.bar(people[j],
                fruit[i][j],
                color=colores[i],
                label=frname[i],
                width=0.5
                )
        # print("{} {} {}".format(people[j], frname[i], fruit[i][j]))

plt.title('Number of Fruit per Person')
plt.xlabel('Quantity of Fruit')
plt.yticks(range(0, 81, 10))
plt.show()
