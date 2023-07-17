#!/usr/bin/env python3
"""
3-upcoming.py
"""
import requests


def maximus(fechas):
    """Function that returns the argmax

    Args:
        fechas (list): list with the dates to be analized

    Returns:
        int: argmax
    """
    fecha = float('-inf')
    retorno = float('-inf')
    for i, f in enumerate(fechas):
        if f > fecha:
            fecha = f
            retorno = i
    return retorno


def main():
    """ main function """
    launch = 'https://api.spacexdata.com/latest/launches/'
    rocket = 'https://api.spacexdata.com/latest/rockets/'
    pad = 'https://api.spacexdata.com/latest/launchpads/'

    r_launch = requests.get(launch)
    r_rocket = requests.get(rocket)
    r_pad = requests.get(pad)

    lista = r_launch.json()
    launch_fechas = [x['date_unix'] for x in lista]
    launch_name = [x['name'] for x in lista]
    launch_rocketId = [x['rocket'] for x in lista]
    lauch_padId = [x['launchpad'] for x in lista]

    max = maximus(launch_fechas)
    rockId = launch_rocketId[max]
    padId = lauch_padId[max]

    max_launch_name = [launch_name[max]]
    max_date = [lista[max]['date_local']]
    max_rocket = [x['name'] for x in r_rocket.json() if x['id'] == rockId]
    max_pad_name = [x['name'] for x in r_pad.json() if x['id'] == padId]
    max_pad_loc = [x['locality'] for x in r_pad.json() if x['id'] == padId]

    im = max_launch_name + max_date + max_rocket + max_pad_name + max_pad_loc

    print('{} ({}) {} - {} ({})'.format(
        im[0], im[1], im[2], im[3], im[4]
    ))


if __name__ == '__main__':
    main()
