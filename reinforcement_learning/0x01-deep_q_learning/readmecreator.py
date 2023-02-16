#!/usr/bin/env python3
"""Modulus Readme Creator"""
import os
import re


"""
Debe tener: 
- Nombre del proyecto - Es el primer párrafo
- Objetivos de aprendizaje
- Tarea - Tabla archivo - Descripción
- About me y datos de contacto
"""

titulo = ''
objetivos = []
tareas = []
description = []
archivos = []

archi = re.compile(r'File:*')
tarea = re.compile(r'\d*\.\s\w*')
iniob = re.compile(r'General\n*')
finob = re.compile(r'Requirements')

with open('origen', 'r', encoding='utf-8') as f:
    lines = f.readlines()

inifin = 0

for i, line in enumerate(lines):
    titulo = lines[0]
    if lines[i] == 'General\n' and lines[i - 1] != 'Requirements\n':
        inifin = 1
        print("Valor de control {}, linea: {}".format(inifin, line))
    if lines[i] == 'General\n' and lines[i - 1] == 'Requirements\n':
        inifin = 0
        print("Valor de control {}, linea: {}".format(inifin, line))
    if inifin == 1:
        objetivos.append(line)
    if tarea.match(line):
        tareas.append(line)
        description.append(lines[i + 3])
    if archi.match(line):
        archivos.append(line.split()[1])

titulo = titulo.replace('\n', '')
for i, o in enumerate(objetivos):
    objetivos[i] = o.replace('\n', '')
for i, t in enumerate(tareas):
    tareas[i] = t.replace('\n', '')
for i, d in enumerate(description):
    description[i] = d.replace(':\n', '')

# print(lines)
# print(titulo)
# print(objetivos)
# print(tareas)
# print(description)
# print(archivos)

unica = []

unica.append('# ' + titulo + ' #\n\n')

unica.append('<img src="https://github.com/RayBar72/holbertonschool-machine_learning/blob/master/image.png" width="1000" height="450">\n\n')

unica.append('## ' + 'Learning Objectives' + ' ##\n\n')

for o in objetivos:
    unica.append('- '+ o + '\n')

unica.append('\n## ' + 'Content Table' + ' ##\n\n')
unica.append('| Task | Description | File |\n')
unica.append('| ----------- | ----------- | ----------- |\n')

for x, y, z in zip(tareas, description, archivos):
    unica.append('| ' + x +' | ' + y + ' | ' + z + ' |\n')

unica.append('\n## ' + 'Authors:' + ' ##\n\n')
unica.append('**Solution by:** Raymundo Barrera Flores. [rbarreraf72@gmail.com](rbarreraf72@gmail.com)')
unica.append('[<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white"/>](https://www.linkedin.com/in/raymundo-barrera-flores-a13022222/\n\n')
unica.append('\n**Project Required by**: HolbertonSchool\n')
# print(unica)

with open('README.md', 'w') as f:
    for u in unica:
        f.write(u)
