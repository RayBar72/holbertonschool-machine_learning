#!/usr/bin/env python3
"""
1. Loop
Modulus that conatins the function loop
"""


end = ['exit', 'quit', 'goodbye', 'bye']

while True:
    question = input('Q: ').lower()
    if question in end:
        print('A: Goodbye')
        break
    else:
        print('A: ')
