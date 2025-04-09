import numpy as np

# Classe Backsolve
class backsolve:
    def __init__(self, A, b):
        self.A = A 
        self.b = b
        print("A: \n", self.A)
        print("\n b: \n", self.b)

        n = len(self.A) # Armazena tamanho de n com base na quantidade de linhas da matriz

        # Armazenar dimensões de A e b
        dimensao_a = A.shape
        dimensao_b = b.shape
        print("\n Dimensao A: ", dimensao_a, "Dimensao B: ", dimensao_b)

        # Verificar se A é uma matriz quadrada
        if dimensao_a[0] != dimensao_a[1]:
            print("\n A não é uma matriz quadrada. Utilize uma matriz quadrada para A.")
            exit()
        
        #Verificar se existe o produto entre A e b
        if dimensao_a[1] != dimensao_b[0]: 
            print("\n A e b não possui dimensões compatíveis para o produto.")
            exit()

        # Verificação se a matriz é triangular superior
        for j in range(n):
            for i in range(j + 1, n):
                if A[i][j] != 0:
                    raise ValueError("Não é uma matriz triangular superior.")

        # Verifica se algum elemento da diagonal principal é zero        
        for i in range(n):
            if A[i][i] == 0:
                raise Exception(f"Elemento nulo na diagonal principal na posição {i+1},{i+1}. Certfique-se que a diagonal principal da matriz A não possua valor 0.")
            
        x = np.zeros(n) # Defini um valor de x qualquer (zero) com base no tamanho da matriz A.
       
        for i in range(n - 1, 0-1, -1): # Iteração em i de trás para frente (inicia no maior valor de i)
            print(f"-------Iteração-------")
            print("i = ", i)
            print("b = ", self.b[i])
            print("A = ", self.A[i][i])
            
            sum_ax = 0 # Define variável de soma = 0
            for j in range(i + 1, n): # Não há calculo de j na primeira interação
                print("j = ", i+ 1)
                
                sum_ax += self.A[i, j] * x[j]
                print(self.A[i, j], "|" ,x[j])
                print("sum_ax = ", sum_ax)
            
        
            x[i] = (self.b[i] - sum_ax) / self.A[i,i]
            print(f"x[{i+1},1] = ", x[i])
            print(x)

# Exemplo de execução
A = np.array([
    [1, 2, 6, 0],
    [0, 1, 6, 1],
    [0, 0, 1, 1],
    [0, 0, 0, 1]
])

b = np.array([
    [1],
    [4],
    [1],
    [1]
])

backsubstitution_obj = backsolve(A, b)
