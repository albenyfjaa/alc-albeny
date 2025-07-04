import numpy as np # Importa a biblioteca NumPy.
                   # Permissão estrita: apenas numpy.array e numpy.sqrt são permitidos.
                   # Portanto, produtos escalares e determinantes são implementados manualmente.

def _produto_escalar(vetor1: np.array, vetor2: np.array) -> float:
    """
    Calcula o produto escalar de dois vetores NumPy 1D manualmente.

    Args:
        vetor1 (np.array): O primeiro vetor.
        vetor2 (np.array): O segundo vetor.

    Returns:
        float: O produto escalar dos dois vetores.

    Raises:
        ValueError: Se os vetores não tiverem a mesma dimensão.
    """
    if vetor1.shape != vetor2.shape:
        raise ValueError("Vetores devem ter a mesma dimensão para o produto escalar.")

    # Soma manual dos produtos dos elementos correspondentes.
    # Adere estritamente à regra de não usar np.sum diretamente.
    return sum(vetor1[i] * vetor2[i] for i in range(len(vetor1)))


def volume_vetores(v1_input: list, v2_input: list, v3_input: list) -> float:
    """
    Calcula o volume de um tetraedro formado por 3 vetores no espaço n-dimensional e a origem.
    A função recebe os vetores como listas (ou iteráveis compatíveis) e os converte
    para arrays NumPy internamente usando np.array().

    O volume é calculado usando a raiz quadrada do determinante da Matriz de Gram.
    Para 3 vetores (formando um tetraedro), a fórmula é V = (1/3!) * sqrt(det(G)).

    Args:
        v1_input (list): O primeiro vetor (pode ser uma lista Python ou similar).
        v2_input (list): O segundo vetor (pode ser uma lista Python ou similar).
        v3_input (list): O terceiro vetor (pode ser uma lista Python ou similar).

    Returns:
        float: O volume do tetraedro.

    Raises:
        TypeError: Se alguma entrada não puder ser convertida para um array NumPy.
        ValueError: Se os vetores não forem 1-dimensionais após a conversão,
                    tiverem dimensões inconsistentes, ou a dimensão for menor que 3.
    """
    # Converter vetores de entrada para arrays NumPy.
    # Isso também lida corretamente com entradas que já são np.array.
    try:
        v1 = np.array(v1_input, dtype=float)
        v2 = np.array(v2_input, dtype=float)
        v3 = np.array(v3_input, dtype=float)
    except Exception as e:
        raise TypeError(f"Não foi possível converter a entrada para um array NumPy. Certifique-se de que as entradas são sequências numéricas válidas. Erro: {e}")

    # Verificar se os arrays são 1-dimensionais após a conversão.
    if not (v1.ndim == 1 and v2.ndim == 1 and v3.ndim == 1):
        raise ValueError("Os vetores convertidos devem ser 1-dimensionais.")

    # Validar que todos os vetores têm a mesma dimensão.
    if not (v1.shape == v2.shape == v3.shape):
        raise ValueError("Todos os vetores devem ter a mesma dimensão após a conversão.")

    dimensao_n = v1.shape[0] # Pega a dimensão N dos vetores

    # Um tetraedro não degenerado requer pelo menos 3 dimensões.
    if dimensao_n < 3:
        raise ValueError(f"Não é possível formar um tetraedro não degenerado com vetores de {dimensao_n} dimensões. A dimensão mínima é 3.")

    # Calcular os produtos escalares para formar a Matriz de Gram (G), que será 3x3.
    g11 = _produto_escalar(v1, v1)
    g12 = _produto_escalar(v1, v2)
    g13 = _produto_escalar(v1, v3)

    g21 = _produto_escalar(v2, v1) # g21 é igual a g12 (Matriz de Gram é simétrica)
    g22 = _produto_escalar(v2, v2)
    g23 = _produto_escalar(v2, v3)

    g31 = _produto_escalar(v3, v1) # g31 é igual a g13
    g32 = _produto_escalar(v3, v2) # g32 é igual a g23
    g33 = _produto_escalar(v3, v3)

    # Calcular o determinante da Matriz de Gram 3x3 manualmente.
    # G = [[g11, g12, g13],
    #      [g21, g22, g23],
    #      [g31, g32, g33]]
    # det(G) = g11*(g22*g33 - g23*g32) - g12*(g21*g33 - g23*g31) + g13*(g21*g32 - g22*g31)

    determinante_gram = (g11 * (g22 * g33 - g23 * g32) -
                         g12 * (g21 * g33 - g23 * g31) +
                         g13 * (g21 * g32 - g22 * g31))

    # O argumento da raiz quadrada não pode ser negativo.
    # Pequenas imprecisões de ponto flutuante podem resultar em valores ligeiramente negativos.
    if determinante_gram < 0:
        # Se o determinante for ligeiramente negativo (muito próximo de zero) devido a erros de precisão,
        # consideramos que o volume é zero (caso de vetores linearmente dependentes).
        volume_final = 0.0
    else:
        volume_final = (1.0 / 6.0) * np.sqrt(determinante_gram)

    return volume_final