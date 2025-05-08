import numpy as np

def norma_p_matriz_2por2(matriz, p):
    """
    Estima a norma-p induzida de uma matriz 2x2.
    """

    # Verificação de tipo e forma
    if not isinstance(matriz, np.ndarray):
        matriz = np.array(matriz, dtype=float)
    if matriz.shape != (2, 2):
        raise ValueError("A matriz deve ser 2x2.")

    # Casos exatos para p = 1 e p = infinito
    if p == 1:
        col_0 = abs(matriz[0,0]) + abs(matriz[1,0])
        col_1 = abs(matriz[0,1]) + abs(matriz[1,1])
        return col_0 if col_0 > col_1 else col_1

    if p == float('inf'):
        linha_0 = abs(matriz[0,0]) + abs(matriz[0,1])
        linha_1 = abs(matriz[1,0]) + abs(matriz[1,1])
        return linha_0 if linha_0 > linha_1 else linha_1

    # Caso especial: p = 2 (norma espectral via autovalores de AᵗA)
    if p == 2:
        a, b = matriz[0,0], matriz[0,1]
        c, d = matriz[1,0], matriz[1,1]
        m11 = a*a + c*c
        m12 = a*b + c*d
        m22 = b*b + d*d

        tr = m11 + m22
        det = m11 * m22 - m12 * m12
        delta = tr**2 - 4 * det
        if delta < 1e-12:
            delta = 0.0
        raiz = delta ** 0.5
        lambda_max = (tr + raiz) / 2.0
        if lambda_max < 0:
            lambda_max = 0.0
        return lambda_max ** 0.5

    # Para outros valores de p ≥ 1: estimativa via amostragem
    def norma_p_vetor(v, p):
        return (abs(v[0])**p + abs(v[1])**p) ** (1/p)

    maior_valor = 0
    N = 20000
    for _ in range(N):
        x = [np.random.randn(), np.random.randn()]
        norma_x = norma_p_vetor(x, p)
        if norma_x < 1e-9:
            continue

        # Produto Ax manual
        Ax = [matriz[0,0]*x[0] + matriz[0,1]*x[1],
              matriz[1,0]*x[0] + matriz[1,1]*x[1]]

        norma_Ax = norma_p_vetor(Ax, p)
        atual = norma_Ax / norma_x
        if atual > maior_valor:
            maior_valor = atual

    return maior_valor
