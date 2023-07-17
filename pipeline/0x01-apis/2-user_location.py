#!/usr/bin/env python3
"""
2-user_location.py
"""
import requests
import sys
import time


def main():
    """
    Main function that using the GitHub API, prints the location of a
    user
    """
    headers = {'Accept': 'application/vnd.github.v3+json'}
    req = requests.get(sys.argv[1], headers=headers)

    if req.status_code == 200:
        print(req.json()['location'])

    elif req.status_code == 404:
        print('Not found')

    elif req.status_code == 403:
        limit = int(req.headers['X-Ratelimit-Reset'])
        now = int(time.time())
        remain = int((limit - now) / 60)
        print('Reset in {} min'.format(remain))
