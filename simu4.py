import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Equação diferencial do pêndulo de Kapitza (adimensional)
def kapitza_ode(s, y, Q, epsilon, nu):
    theta, dtheta = y
    ddtheta = -Q*dtheta - (1 - epsilon * nu**2 * np.cos(nu * s)) * np.sin(theta)
    return [dtheta, ddtheta]

# --- PARÂMETROS CONSAGRADOS PARA CAOS ---
Q = 0.05        # Amortecimento muito baixo
nu = 11.0       # Frequência moderada (não é >> 1)
eps = 0.28      # Amplitude de excitação alta
# Forçante inercial eps * nu^2 approx 33.9

t_max = 500     # Simulação longa para o atrator se revelar
t_eval = np.linspace(0, t_max, 100000) # Alta resolução
y0 = [1.0, 0]   # Condição inicial: afastado da vertical

# Executar a simulação
sol_strange = solve_ivp(kapitza_ode, (0, t_max), y0, args=(Q, eps, nu), t_eval=t_eval, method='RK45')

# Ajuste do ângulo para o intervalo [-pi, pi]
theta_strange = (sol_strange.y[0] + np.pi) % (2 * np.pi) - np.pi

# --- PLOTAGEM DO ATRATOR ESTRANHO ---
plt.figure(figsize=(8, 8))
plt.plot(theta_strange, sol_strange.y[1], lw=0.2, color='#4a148c', alpha=1)
plt.title(f'Atrator Estranho: Caos Deterministico\n$\\epsilon={eps}, \\nu={nu}$', fontsize=12)
plt.xlabel('Ângulo $\\theta$ (rad)', fontsize=10)
plt.ylabel('Velocidade Angular $\\dot{\\theta}$', fontsize=10)
plt.xlim(-np.pi, np.pi)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('strange_attractor.png', dpi=300) # Alta resolução
plt.show()
