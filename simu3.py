import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def kapitza_ode(s, y, Q, epsilon, nu):
    theta, dtheta = y
    ddtheta = -Q*dtheta - (1 - epsilon * nu**2 * np.cos(nu * s)) * np.sin(theta)
    return [dtheta, ddtheta]

t_max = 100
t_eval = np.linspace(0, t_max, 10000)

# 1. Ressonância Paramétrica (Frequência baixa, instabiliza o repouso)
sol_res = solve_ivp(kapitza_ode, (0, t_max), [0.1, 0], args=(0.05, 0.4, 2.0), t_eval=t_eval)

# 3. Regime de Batimento/Intermediário (Transição para o Caos)
sol_trans = solve_ivp(kapitza_ode, (0, t_max), [1.0, 0], args=(0.01, 0.1, 15), t_eval=t_eval)

# Plotagem
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

axs[0].plot(sol_res.t, sol_res.y[0], color='forestgreen')
axs[0].set_title('Ressonância Paramétrica (Instabilização de $\\theta=0$ em $\\nu \\approx 2$)')
axs[0].set_ylabel('$\\theta$ (rad)')


axs[1].plot(sol_trans.t, sol_trans.y[0], color='purple')
axs[1].set_title('Regime de Transição (Oscilações complexas de baixa frequência)')
axs[1].set_ylabel('$\\theta$ (rad)')
axs[1].set_xlabel('Tempo adimensional (s)')

plt.tight_layout()
plt.show()
