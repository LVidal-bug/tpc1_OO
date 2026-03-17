import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def kapitza_ode(s, y, Q, epsilon, nu):
    theta, dtheta = y
    # Equação original adimensional
    ddtheta = -Q*dtheta - (1 - epsilon * nu**2 * np.cos(nu * s)) * np.sin(theta)
    return [dtheta, ddtheta]

# Parâmetros que satisfazem a estabilidade (eps * nu > sqrt(2))
Q = 0.2         # Amortecimento para convergência visível
eps = 0.05      
nu = 40         # eps * nu = 2.0 (Estável, pois 2.0 > 1.41)
t_max = 100
t_eval = np.linspace(0, t_max, 10000)

# 5 Condições iniciais razoáveis [theta0, dtheta0]
# Variando desde perto do topo até ângulos mais abertos
condicoes_iniciais = [
    [np.pi - 0.2, 0],   # Muito perto do topo
    [np.pi + 0.4, 0],   # Do outro lado
    [np.pi - 0.8, 0],   # Ângulo mais aberto (~45 graus do topo)
    [np.pi, 0.5],       # Exatamente no topo mas com velocidade inicial
    [np.pi - 0.5, -0.3] # Longe e movendo-se para longe
]

plt.figure(figsize=(12, 6))

for i, y0 in enumerate(condicoes_iniciais):
    sol = solve_ivp(kapitza_ode, (0, t_max), y0, args=(Q, eps, nu), t_eval=t_eval, method='RK45')
    plt.plot(sol.t, sol.y[0], label=f'Caso {i+1}: $\\theta_0={y0[0]:.2f}$ e v0 = {y0[1]} m/s')

plt.axhline(y=np.pi, color='black', linestyle='--', alpha=0.5, label='Equilíbrio $\\theta = \pi$')
plt.title(f'Convergência para o Topo (Estabilidade Garantida: $\epsilon\\nu = {eps*nu}$)', fontsize=14)
plt.xlabel('Tempo adimensional ($s$)', fontsize=12)
plt.ylabel('Ângulo $\\theta$ (rad)', fontsize=12)
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()