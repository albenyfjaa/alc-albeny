import numpy as np

def analise_posto_norma(n_valor):
    for n in n_valor:
        u = np.random.rand(n, 1)
        # print("u:", u)
        v = np.random.rand(n, 1)
        # print("v:", v)
        A = u @ v.T  # Produto externo
        
        rank = np.linalg.matrix_rank(A)
        norm_u = np.linalg.norm(u, 2)
        norm_v = np.linalg.norm(v, 2)
        norm_A = np.linalg.norm(A, 2)  # Norma 2 (espectral)

        print(f"n = {n}")
        print(f"Posto de A = {rank}")
        print(f"||u||_2 = {norm_u}")
        print(f"||v||_2 = {norm_v}")    
        print(f"||A||_2 = {norm_A}")
        print(f"||u||_2 * ||v||_2 = {(norm_u*norm_v)}")
        print("-" * 40)

# Teste com diferentes dimensoes n (5, 15, 25)
n_dimensoes = [5, 15, 25]

# Execucao
analise_posto_norma(n_dimensoes)