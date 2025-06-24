import numpy as np
import scipy.linalg

def lu_decomp_partial_pivoting(A):
    """
    Decomposicao LU com pivotamento parcial (Algoritmo de Gauss).
    Retorna L, U, P tal que PA = LU.
    """
    A = A.copy().astype(float)
    n = A.shape[0]
    L = np.eye(n)
    P = np.eye(n)

    for i in range(n):
        pivot_index = np.argmax(np.abs(A[i:, i])) + i
        if pivot_index != i:
            A[[i, pivot_index]] = A[[pivot_index, i]]
            P[[i, pivot_index]] = P[[pivot_index, i]]
            if i > 0:
                L[[i, pivot_index], :i] = L[[pivot_index, i], :i]
        for j in range(i+1, n):
            L[j, i] = A[j, i] / A[i, i]
            A[j] -= L[j, i] * A[i]
    U = np.triu(A)
    return L, U, P

# === MATRIZ 3x3 ===
A1 = np.array([
    [2, 3, 1],
    [4, 7, 2],
    [6, 18, -1]
], dtype=float)

# === MATRIZ 4x4 ===
A2 = np.array([
    [14.,  4.,  5.,  3.],
    [ 4., 13.,  8.,  3.],
    [ 2.,  1.,  2., 11.],
    [ 1., 13.,  6.,  3.]
], dtype=float)

# === Funcao para exibir os resultados de uma matriz ===
def mostrar_resultados(A, nome):
    print("\n" + "="*10 + f" {nome} " + "="*10)
    print("Matriz A:")
    print(A)

    # Implementacao propria
    L1, U1, P1 = lu_decomp_partial_pivoting(A)
    print("\n--- Implementacao Propria ---")
    print("P =\n", P1)
    print("L =\n", L1)
    print("U =\n", U1)

    # Funcao SciPy
    P2, L2, U2 = scipy.linalg.lu(A)
    print("\n--- SciPy ---")
    print("P =\n", P2)
    print("L =\n", L2)
    print("U =\n", U2)

    # Validacao
    print("\nValidacao:")
    print("P @ A aproximadamente igual a L @ U (implementacao)?", np.allclose(P1 @ A, L1 @ U1))
    print("P @ A ≈ L @ U (SciPy)?", np.allclose(np.dot(np.linalg.inv(P2), A), np.dot(L2, U2)))


# Mostrar os resultados para ambas as matrizes
mostrar_resultados(A1, "Matriz 3x3")
mostrar_resultados(A2, "Matriz 4x4")