#!/usr/bin/env python3
"""
0-passengers.py
"""
import requests


def availableShips(passengerCount):
    """Using the Swapi API, create a method that returns the list of
    ships that can hold a given number of passenger

    Args:
        passengerCount (int): Number of passangers to be carried

    Returns:
        list: ships avaiable, if not empty list
    """
    resultados = []
    if type(passengerCount) is not int:
        return resultados
    url = 'https://swapi-api.hbtn.io/api/starships/'
    while url:
        resp = requests.get(url)
        resp_dict = resp.json()
        records = resp_dict['results']
        for record in records:
            try:
                passa = int(record['passengers'])
            except Exception as e:
                passa = float('-inf')
            if passa >= passengerCount:
                resultados.append(record['name'])
        url = resp_dict['next']
    return resultados
