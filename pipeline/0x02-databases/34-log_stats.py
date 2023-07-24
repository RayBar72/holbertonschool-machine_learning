#!/usr/bin/env python3
"""
log_stats
"""
from pymongo import MongoClient


def main():
    """
    Main function
    """
    client = MongoClient()
    nginx = client.logs.nginx
    metodos = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']

    print('{} logs'.format(nginx.count_documents({})))
    print('Methods:')
    for m in metodos:
        print('\tmethod {}: {}'.format(m,
                                       nginx.count_documents({'method': m})))
    print('{} status check'.format(nginx.count_documents({'method': 'GET',
                                                          'path': '/status'})))


if __name__ == '__main__':
    main()
