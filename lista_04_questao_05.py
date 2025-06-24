import numpy as np

def modified_gram_schmidt(A):
    """
    Implementa o Algoritmo 14.3 - Decomposicao QR usando Gram-Schmidt Modificado.
    
    Parametros:
    A : numpy.ndarray (m x n)
        Matriz de entrada (com m linhas e n colunas)

    Retorna:
    Q : numpy.ndarray (m x n)
        Matriz com colunas ortonormais
    R : numpy.ndarray (n x n)
        Matriz triangular superior
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for i in range(n):
        # Inicialmente, q_i recebe a i-esima coluna de A
        Q[:, i] = A[:, i]

        for j in range(i):
            # Projecao de Q[:, i] sobre Q[:, j]
            R[j, i] = np.dot(Q[:, j], Q[:, i])
            Q[:, i] -= R[j, i] * Q[:, j]

        # Normaliza Q[:, i]
        R[i, i] = np.linalg.norm(Q[:, i])
        Q[:, i] = Q[:, i] / R[i, i]

    return Q, R

# Matriz do Exercicio 14.15
A = np.array([
    [1, 9, 0, 5, 3, 2],
    [-6, 3, 8, 2, -8, 0],
    [3, 15, 23, 2, 1, 7],
    [3, 57, 35, 1, 7, 9],
    [3, 5, 6, 15, 55, 2],
    [33, 7, 5, 3, 5, 7]
], dtype=float)

Q, R = modified_gram_schmidt(A)

print("Q =\n", Q)
print("R =\n", R)
print("Reconstrucao A aproximadamente igual a Q @ R?\n", np.allclose(A, Q @ R))

# Decomposicao QR usando NumPy
Q, R = np.linalg.qr(A)

# Impressao dos resultados
print("Q (numpy.linalg.qr) =\n", Q)
print("\nR (numpy.linalg.qr) =\n", R)

# Verificacao: A aproximadamente igual a QR?
print("\nReconstrucao A aproximadamente igual a Q @ R?\n", np.allclose(A, Q @ R))