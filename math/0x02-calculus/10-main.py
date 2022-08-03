#!/usr/bin/env python3

poly_derivative = __import__('10-matisse').poly_derivative

# No list
poly = 'xyz'
print(poly_derivative(poly))

# No poly
poly = []
print(poly_derivative(poly))

# None in list
poly = [1, 2, 3, 4, None]
print(poly_derivative(poly))

# Largo 1
poly = [1]
print(poly_derivative(poly))

# con letras
poly = ['xyz']
print(poly_derivative(poly))

# con letras
poly = ['xyz', 1, 2, 3, 4]
print(poly_derivative(poly))

print('----------------------------------------------------------------')

# prueba
poly = [8, 1, 10, 0, 12, 0, 14, 15, 16]
print(poly_derivative(poly))