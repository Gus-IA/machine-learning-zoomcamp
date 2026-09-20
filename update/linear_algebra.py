import numpy as np

u = np.array([2, 4, 5, 6])

v = np.array([1, 0, 0, 2])


# multiplicación de dos vectores
def vector_vector_multiplication(u, v):
    assert u.shape[0] == v.shape[0]

    n = u.shape[0]

    result = 0.0

    for i in range(n):
        result = result + u[i] * v[i]

    return result


print(vector_vector_multiplication(u, v))

print(u.dot(v))

U = np.array([[2, 4, 5, 6], [1, 2, 1, 2], [3, 1, 2, 1]])


# multiplicación de un vector y una matriz
def matrix_vector_multiplication(U, v):
    assert U.shape[1] == v.shape[0]

    num_rows = U.shape[0]

    result = np.zeros(num_rows)

    for i in range(num_rows):
        result[i] = vector_vector_multiplication(U[i], v)

    return result


print(matrix_vector_multiplication(U, v))

print(U.dot(v))


V = np.array(
    [
        [1, 1, 2],
        [0, 0.5, 1],
        [0, 2, 1],
        [2, 1, 0],
    ]
)


# multiplicar dos matrices
def matrix_matrix_multiplication(U, V):
    assert U.shape[1] == V.shape[0]

    num_rows = U.shape[0]
    num_cols = V.shape[1]

    result = np.zeros((num_rows, num_cols))

    for i in range(num_cols):
        vi = V[:, i]
        Uvi = matrix_vector_multiplication(U, vi)
        result[:, i] = Uvi

    return result


print(matrix_matrix_multiplication(U, V))

print(U.dot(V))

# identificar matriz
I = np.eye(3)
print(I)

print(V.dot(I))

# matriz inverse
Vs = V[[0, 1, 2]]
print(Vs)

Vs_inv = np.linalg.inv(Vs)
print(Vs_inv)

print(Vs_inv.dot(Vs))
