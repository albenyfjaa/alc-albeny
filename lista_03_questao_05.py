import matplotlib.pyplot as plt
import numpy as np

def wilkinson_bidiagonal(n):
    A = np.diag(np.arange(n, 0, -1, dtype=float))  # força tipo float
    A += np.diag([n] * (n - 1), k=1)
    return A

n_values = range(1, 16)
cond_numbers = [np.linalg.cond(wilkinson_bidiagonal(n)) for n in n_values]

plt.figure(figsize=(8, 5))
plt.plot(n_values, cond_numbers, marker='o')
plt.title("Número de condição da matriz bidiagonal de Wilkinson")
plt.xlabel("Ordem n")
plt.ylabel("Número de condição")
plt.grid(True)
plt.show()

# Matriz original A(20)
n = 20
A = wilkinson_bidiagonal(n)

# Autovalores da matriz original
eigvals_original = np.linalg.eigvals(A)

# Criar perturbação pequena
np.random.seed(0)  # garante reprodutibilidade
perturbation = 1e-10 * np.random.randn(n, n)
A_perturbed = A + perturbation

# Autovalores da matriz perturbada
eigvals_perturbed = np.linalg.eigvals(A_perturbed)

# Imprimir os autovalores
print("Autovalores da matriz original:\n", np.sort(np.real(eigvals_original)))
print("\nAutovalores da matriz perturbada:\n", np.sort(np.real(eigvals_perturbed)))

# Gráfico comparativo
plt.figure(figsize=(9, 5))
plt.plot(np.sort(np.real(eigvals_original)), 'o-', label='Original')
plt.plot(np.sort(np.real(eigvals_perturbed)), 'x--', label='Perturbada (1e-10)')
plt.title("Autovalores da matriz bidiagonal de Wilkinson (n=20)")
plt.xlabel("Índice ordenado")
plt.ylabel("Autovalores (parte real)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()