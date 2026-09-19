import numpy as np

# 5 zeros
print(np.zeros(5))

# 10 veces valor de 2.5
print(np.full(10, 2.5))

# array numpy
a = np.array([1, 2, 3, 5, 7, 12])
print(a)

# posición 3 del array
print(a[2])

# modificar la posición 3
a[2] = 10
print(a)

# rango del 3 al 10 -1
print(np.arange(3, 10))

# crea 11 entre 0 y 1
print(np.linspace(0, 1, 11))

# array multidimensional de 5 dimensiones y 2 valores
print(np.zeros((5, 2)))

n = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(n)

print(n[0, 1])

# array random entre 0 y 1
np.random.seed(2)  # tienes el mismo random
print(np.random.rand(5, 2))

np.random.seed(2)
print(np.random.randn(5, 2))  # randn negativo

# comprobar si en el array a hay valores más grandes o iguales a 2
print(a >= 2)

# numero más pequeño
print(a.min())

# número más grande
print(a.max())

# suma de los valores
print(a.sum())
