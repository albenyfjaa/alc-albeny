import numpy as np

# Função Posição Efetuador Final
def posicao_efetuador(theta1_graus, theta2_graus, L1=20.0, L2=15.0):
    
    # Verifica se angulos fornecidos estao entre 0 e 360 graus
    if not 0 <= theta1_graus <= 360:
        raise ValueError("O angulo theta1_graus deve estar entre 0 e 360 graus.")

    if not 0 <= theta2_graus <= 360:
        raise ValueError("O angulo theta2_graus deve estar entre 0 e 360 graus.")

    # Converte angulos fornecidos para radianos
    theta1 = np.radians(theta1_graus)
    theta2 = np.radians(theta2_graus)
      
    # Calcular os angulos e coordenadas
    theta_total = theta1 + theta2
    XU = round(L1 * np.cos(theta1) + L2 * np.cos(theta_total), 1)
    YU = round(L1 * np.sin(theta1) + L2 * np.sin(theta_total), 1)

    print("Posiçao do efetuador final XU:", XU, "cm",  "Posiçao do efetuador final YU:", YU,"cm")

    return XU, YU

# Exemplo de execução
posicao_efetuador(90, 0)
