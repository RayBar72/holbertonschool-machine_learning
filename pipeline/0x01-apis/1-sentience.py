#!/usr/bin/env python3
"""
1-sentience.py
"""
import requests


def sentientPlanets():
    """Using the Swapi API, create a method that returns the list
    of names of the home planets of all sentient specie

    Returns:
        list: Home planets of sentient species
    """
    resultados = []
    url = 'https://swapi-api.alx-tools.com/api/species/'
    while url:
        resp = requests.get(url)
        resp_dict = resp.json()
        records = resp_dict['results']
        for record in records:
            if record['designation'] == 'sentient':
                resultados.append([record['designation'],
                                   record['name'], record['homeworld']])
        url = resp_dict['next']
    respuesta = []
    for i in resultados:
        if i[2] is not None:
            i = requests.get(i[2])
            i = i.json()['name']
            respuesta.append(i)
    return respuesta
