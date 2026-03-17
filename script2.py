import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def kapitza_ode(s, y, Q, epsilon, nu):
    theta, dtheta = y
    ddtheta = -Q*dtheta - (1 - epsilon * nu**2 * np.cos(nu * s)) * np.sin(theta)
    return [dtheta, ddtheta]

# Parâmetros
Q = 0.1
target_eff = 2.0  # (eps * nu)^2 / 2 = 2.0 -> Estável pela teoria

# CASO 1: Alta frequência (Teoria de Kapitza funciona)
nu1 = 50
eps1 = np.sqrt(2 * target_eff) / nu1 # eps1 approx 0.04

# CASO 2: Baixa frequência (Teoria de Kapitza falha)
nu2 = 2
eps2 = np.sqrt(2 * target_eff) / nu2 # eps2 approx 0.4

t_max = 60
t_eval = np.linspace(0, t_max, 10000)
y0 = [np.pi - 0.1, 0] # Perto do topo

# Simulações
sol1 = solve_ivp(kapitza_ode, (0, t_max), y0, args=(Q, eps1, nu1), t_eval=t_eval)
sol2 = solve_ivp(kapitza_ode, (0, t_max), y0, args=(Q, eps2, nu2), t_eval=t_eval)

# Gráficos
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(sol1.t, sol1.y[0], color='blue')
ax1.axhline(y=np.pi, color='r', linestyle='--')
ax1.set_title(f'Sucesso da Teoria: $\\nu={nu1}$ (Alta Frequência) e $\\epsilon={eps1}$ (Baixa Amplitude)' )
ax1.set_ylabel('$\\theta$ (rad)')

ax2.plot(sol2.t, sol2.y[0], color='orange')
ax2.axhline(y=np.pi, color='r', linestyle='--')
ax2.set_title(f'Falha da Teoria: $\\nu={nu2}$ (Baixa Frequência) e $\\epsilon={eps2}$ (Alta Amplitude)')
ax2.set_ylabel('$\\theta$ (rad)')
ax2.set_xlabel('Tempo adimensional (s)')

plt.tight_layout()
plt.show()