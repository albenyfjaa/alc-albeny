# Importações permitidas conforme a solicitação
from numpy import array, zeros, abs
from numpy.linalg import eig

def aproximacao_truncada(matriz_original, num_autovalores_manter):
    """
        Cria uma matriz A' que representa uma versão aproximada da matriz de entrada,
        focando nos 'num_autovalores_manter' autovalores de maior módulo.

        Esta versão da função executa as operações de produto e inversão de matrizes
        sem o auxílio direto de bibliotecas externas, atendendo às restrições do problema.

        Argumentos:
            matriz_original (list ou numpy.array): A matriz quadrada inicial (de dimensão n x n).
            num_autovalores_manter (int): A quantidade de autovalores com maior valor absoluto
                                        que serão utilizados na construção da aproximação.

        Retorna:
            numpy.array: A matriz aproximada gerada (n x n).

        Exceções:
            ValueError: Disparada se a matriz fornecida não for quadrada, se o parâmetro
                        'num_autovalores_manter' for inválido, ou se a matriz de autovetores for singular.
    """

    # =========================================================================
    # Funções auxiliares para operações de matriz, implementadas manualmente
    # para evitar funções restritas do numpy.linalg (exceto eig, que é permitida).
    # =========================================================================

    def realizar_multiplicacao_matrizes(mat_a, mat_b):
        """Multiplica duas matrizes (mat_a @ mat_b) sem usar numpy.dot ou @."""
        linhas_a, colunas_a = mat_a.shape
        linhas_b, colunas_b = mat_b.shape

        if colunas_a != linhas_b:
            raise ValueError("As dimensões das matrizes são incompatíveis para multiplicação.")

        matriz_resultado = zeros((linhas_a, colunas_b))
        for idx_l in range(linhas_a):
            for idx_c in range(colunas_b):
                soma_atual = 0
                for k_idx in range(colunas_a):
                    soma_atual += mat_a[idx_l, k_idx] * mat_b[k_idx, idx_c]
                matriz_resultado[idx_l, idx_c] = soma_atual
        return matriz_resultado

    def calcular_inversa_matriz(matriz_entrada):
        """Calcula a inversa de uma matriz quadrada usando o método de eliminação de Gauss-Jordan."""
        dimensao = matriz_entrada.shape[0]
        if matriz_entrada.shape[1] != dimensao:
            raise ValueError("A matriz de entrada deve ser quadrada para ser invertida.")

        # Cria uma matriz identidade da mesma dimensão
        matriz_identidade = zeros((dimensao, dimensao))
        for i in range(dimensao):
            matriz_identidade[i, i] = 1.0
        
        # Forma a matriz aumentada [matriz_entrada | matriz_identidade]
        sistema_aumentado = array(zeros((dimensao, 2 * dimensao)))
        sistema_aumentado[:, :dimensao] = matriz_entrada.copy() # O .copy() é crucial para não modificar a original
        sistema_aumentado[:, dimensao:] = matriz_identidade

        # Aplica a eliminação de Gauss-Jordan para encontrar a inversa
        for i in range(dimensao):
            # Pivoteamento: encontra a linha com o maior pivô para estabilidade numérica
            linha_pivo_max = i
            for j in range(i + 1, dimensao):
                if abs(sistema_aumentado[j, i]) > abs(sistema_aumentado[linha_pivo_max, i]):
                    linha_pivo_max = j
            
            # Troca a linha atual pela linha que contém o maior pivô
            sistema_aumentado[[i, linha_pivo_max]] = sistema_aumentado[[linha_pivo_max, i]]

            # Verifica se a matriz é singular (não invertível)
            if abs(sistema_aumentado[i, i]) < 1e-10: # Usa uma pequena tolerância para pivô próximo de zero
                raise ValueError("Matriz é singular e não pode ser invertida.")

            # Normaliza a linha do pivô (para que o elemento da diagonal seja 1)
            fator_divisao = sistema_aumentado[i, i]
            sistema_aumentado[i, :] /= fator_divisao

            # Zera os outros elementos na coluna do pivô
            for j in range(dimensao):
                if i != j:
                    fator_subtracao = sistema_aumentado[j, i]
                    sistema_aumentado[j, :] -= fator_subtracao * sistema_aumentado[i, :]
        
        # Extrai a parte direita da matriz aumentada, que agora é a inversa
        matriz_inversa = sistema_aumentado[:, dimensao:]
        return matriz_inversa

    
    matriz_numpy = array(matriz_original, dtype=float)

    # --- Diretriz 1: Verificar se a matriz é quadrada ---
    if matriz_numpy.ndim != 2 or matriz_numpy.shape[0] != matriz_numpy.shape[1]:
        raise ValueError("A matriz de entrada deve ser quadrada para o cálculo de autovalores.")

    n = matriz_numpy.shape[0]

    # --- Diretriz 2: Verificar se 'num_autovalores_manter' é um valor válido ---
    if num_autovalores_manter > n:
        raise ValueError(f"O número de autovalores a manter (m={num_autovalores_manter}) não pode ser maior "
                         f"que a dimensão da matriz (n={n}).")
    if num_autovalores_manter < 0:
        raise ValueError("O número de autovalores a manter (m) não pode ser negativo.")

    # --- Passo 1: Realizar a decomposição de autovalores ---
    autovalores, autovetores = eig(matriz_numpy)

    # --- Passo 2: Identificar os 'num_autovalores_manter' autovalores de maior valor absoluto ---
    autovalores_absolutos = abs(autovalores)
    indices_ordenados = autovalores_absolutos.argsort()
    indices_a_preservar = indices_ordenados[-num_autovalores_manter:]

    # --- Passo 3: Construir a nova matriz diagonal de autovalores (D') ---
    matriz_diagonal_modificada = zeros((n, n), dtype=complex) # Usar tipo complexo para consistência
    for idx in indices_a_preservar:
        matriz_diagonal_modificada[idx, idx] = autovalores[idx]

    # --- Passo 4: Reconstruir a matriz aproximada (A') usando as funções manuais ---
    matriz_autovetores = autovetores
    matriz_inversa_autovetores = calcular_inversa_matriz(matriz_autovetores)
    
    # A' = P @ D' @ P⁻¹
    produto_temp = realizar_multiplicacao_matrizes(matriz_autovetores, matriz_diagonal_modificada)
    matriz_aproximada_final = realizar_multiplicacao_matrizes(produto_temp, matriz_inversa_autovetores)
    
    return matriz_aproximada_final.real # Retorna a parte real, conforme autovalores reais