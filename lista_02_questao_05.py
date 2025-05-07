import numpy as np

# ============================
# Função 1: Verifica por definição
# ============================
def is_orthogonal_by_definition(A, tol=1e-5):
    """
    Verifica se uma matriz A é ortogonal pela definição:
    A é ortogonal se A.T @ A = I (matriz identidade)
    
    Parâmetros:
        A: matriz quadrada (numpy array)
        tol: tolerância numérica para comparação
        
    Retorna:
        True se A é ortogonal pela definição, False caso contrário
    """
    n = A.shape[0]
    identidade = np.eye(n)
    return np.allclose(A.T @ A, identidade, atol=tol)

# ============================
# Função 2: Verifica pelas colunas (vetores)
# ============================
def is_orthogonal_by_vectors(A, tol=1e-5):
    """
    Verifica se as colunas de A têm norma 1 e são ortogonais entre si.
    Isto é, testa se:
        - Cada coluna tem norma euclidiana igual a 1
        - O produto interno entre quaisquer duas colunas diferentes é 0
        
    Parâmetros:
        A: matriz quadrada (numpy array)
        tol: tolerância numérica
        
    Retorna:
        True se as colunas forem ortonormais, False caso contrário
    """
    n = A.shape[1]
    for i in range(n):
        norma = np.linalg.norm(A[:, i])
        if not np.isclose(norma, 1.0, atol=tol):
            return False  # coluna não tem norma 1
        for j in range(i+1, n):
            prod = np.dot(A[:, i], A[:, j])
            if not np.isclose(prod, 0.0, atol=tol):
                return False  # colunas não são ortogonais
    return True

# ============================
# Matrizes dos Exercícios 6.38 e 6.39
# ============================
P1_638a = np.array([
    [-0.40825,  0.43644,  0.80178],
    [-0.8165 ,  0.21822, -0.53452],
    [-0.40825, -0.87287,  0.26726]
])

P2_638b = np.array([
    [-0.51450,  0.48507,  0.70711],
    [-0.68599, -0.72761,  0.00000],
    [ 0.51450, -0.48507,  0.70711]
])

P1_639a = np.array([
    [-0.58835,  0.70206,  0.40119],
    [-0.78446, -0.37524, -0.49377],
    [-0.19612, -0.60523,  0.77152]
])

P2_639b = np.array([
    [-0.47624, -0.4264 ,  0.30151],
    [ 0.087932, 0.86603, -0.40825],
    [-0.87491, -0.26112, 0.86164]
])

# ============================
# Testes para cada matriz
# ============================
matrizes = {
    "6.38 a (P1)": P1_638a,
    "6.38 b (P2)": P2_638b,
    "6.39 a (P1)": P1_639a,
    "6.39 b (P2)": P2_639b,
}

# Avaliação
for nome, matriz in matrizes.items():
    by_def = is_orthogonal_by_definition(matriz)
    by_vec = is_orthogonal_by_vectors(matriz)
    print(f"\n{nome}")
    print(f" - Ortogonal pela definição? {'Sim' if by_def else 'Não'}")
    print(f" - Ortogonal pelas colunas?   {'Sim' if by_vec else 'Não'}")
