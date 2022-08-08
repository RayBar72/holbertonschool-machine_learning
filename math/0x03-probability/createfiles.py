#!/usr/bin/env python3
# Script to create files for project

lisa = [
    'poisson.py',
    'exponential.py',
    'normal.py',
    'binomial.py',
]

for i in range(13):
    lisa.append(str(i) + '-main.py')

for x in lisa:
    with open(x, 'w') as f:
        pass
