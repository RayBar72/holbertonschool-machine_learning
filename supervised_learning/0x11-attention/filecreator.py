#!/usr/bin/env python3
import re

lista = []

archivo = re.compile(r'File:*')

with open('README.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    c = archivo.match(line)
    if c:
        rs = line.split()
        lista.append(rs[1])

for i in range(len(lista)):
    lista.append(str(i) + '-main.py')

for x in lista:
    with open(x, 'w') as fi:
        pass
