#!/usr/bin/env python3

poly_integral = __import__('17-integrate').poly_integral

dicky ={1: 0}
poly = [5, 3, 0, 1]
print(poly_integral(poly, None))


# No list
poly = 'xyz'
print(poly_integral(poly))

# No poly
poly = []
print(poly_integral(poly))

# None in list
poly = [1 / 1, 2, 3, 4]
print(poly_integral(poly))

# Largo 1
poly = [1]
print(poly_integral(poly))

# con letras
poly = ['xyz']
print(poly_integral(poly))

# con letras
poly = ['xyz', 1, 2, 3, 4]
print(poly_integral(poly))

print('----------------------------------------------------------------')

# prueba
poly = [8, 1, 10, 0, 12, 0, 14, 15, 16]
print(poly_integral(poly))

poly = [0, 0]
print(poly_integral(poly))
