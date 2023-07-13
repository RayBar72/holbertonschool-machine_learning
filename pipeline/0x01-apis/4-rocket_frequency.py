#!/usr/bin/env python3
"""
4-rocket_frequency.py
"""
import requests


def main():
    """
    Main function that queries the SpaceX API and prints the
    number of launches per rocket.
    """
    url_rockets = 'https://api.spacexdata.com/latest/rockets/'
    resp_rockets = requests.get(url_rockets).json()

    url_launch = 'https://api.spacexdata.com/latest/launches/'
    resp_launc = requests.get(url_launch).json()
    print(len(resp_rockets))

    id_name = {x['id']: x['name'] for x in resp_rockets}
    id_forsum = {x['id']: [] for x in resp_rockets}

    for k_id in id_forsum.keys():
        for i in resp_launc:
            if k_id == i['rocket']:
                id_forsum[k_id].append(1)

    suma = {id_name[k]: sum(v) for k, v in id_forsum.items()}

    sort_name = dict(sorted(suma.items(), key=lambda x: x))
    sort_number = sorted(sort_name.items(), key=lambda x: x[1], reverse=True)
    for x in sort_number:
        print('{}: {}'.format(x[0], x[1]))


if __name__ == '__main__':
    main()
