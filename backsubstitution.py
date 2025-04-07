import numpy as np

# Classe Backsolve
class backsolve:
    def __init__(self, A, b):
        self.A = A 
        self.b = b
        print("A: ", self.A)
        print("b: ", self.b)

        n = len(self.A) # Armazena tamanho de n com base no tamanho da matriz A
        x = np.zeros(n) # Defini um valor de x qualquer (zero) com base no tamanho da matriz A.

        print("n: ", n)
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

