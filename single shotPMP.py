import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root

# Constants
m = 0.250       # mass (kg)
Ixx = 0.01    # moment of inertia (kg*m^2)
g = 9.81      # gravity (m/s^2)
T = 5.0       # total time (s)
N = 500       # number of time steps
dt = T / N
time = np.linspace(0, T, N)

# Target final state (hover at origin)
x_target = np.array([5, 5, 0, 0, 0, 0])

# Initial state: start from (x=5, y=5), flat and stationary
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Define dynamics
def dynamics(t, state, u):
    x, y, th, vx, vy, w = state
    F, tau = u
    dxdt = np.array([
        vx,
        vy,
        w,
        -F / m * np.sin(th),
        F / m * np.cos(th) - g,
        tau / Ixx
    ])
    return dxdt

# Define Hamiltonian
def hamiltonian(x, p, u):
    F, tau = u
    x_, y_, th, vx, vy, w = x
    px, py, pth, pvx, pvy, pw = p
    f = dynamics(0, x, u)
    L = 0.5 * (F**2 + tau**2)  # control effort cost
    return np.dot(p, f) + L

# PMP: given x and p, compute optimal control u that minimizes H
def optimal_control(x, p):
    th = x[2]
    pvx, pvy, pw = p[3], p[4], p[5]
    dH_dF = -pvx * (-np.sin(th) / m) + pvy * (np.cos(th) / m)
    F_opt = -dH_dF
    tau_opt = -pw / Ixx  # since ∂H/∂tau = pw/Ixx + tau
    return np.array([F_opt, tau_opt])

# Simulate with shooting method (guess terminal costate)
def simulate_pmp(pT_guess):
    p = np.zeros((N, 6))
    x = np.zeros((N, 6))
    u = np.zeros((N, 2))
    
    # Terminal costate (assume quadratic terminal cost)
    p[-1] = pT_guess
    
    # Backward integrate costate
    for i in range(N-1, 0, -1):
        x_dummy = np.zeros(6)  # unknown x, approximate around zero
        u_dummy = np.array([0, 0])
        eps = 1e-5
        dH_dx = np.zeros(6)
        for j in range(6):
            x1 = x_dummy.copy()
            x2 = x_dummy.copy()
            x1[j] -= eps
            x2[j] += eps
            H1 = hamiltonian(x1, p[i], u_dummy)
            H2 = hamiltonian(x2, p[i], u_dummy)
            dH_dx[j] = (H2 - H1) / (2 * eps)
        p[i-1] = p[i] + dH_dx * dt

    # Forward simulate state
    x[0] = x0
    for i in range(N-1):
        u[i] = optimal_control(x[i], p[i])
        x[i+1] = x[i] + dynamics(time[i], x[i], u[i]) * dt

    return x, u, p

# Initial costate guess
pT_guess = np.array([0, 0, 0, 0, 0, 0])
x, u, p = simulate_pmp(pT_guess)

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time, x[:, 0], label='x')
plt.plot(time, x[:, 1], label='y')
plt.title('Quadrotor Position')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(time, x[:, 3], label='vx')
plt.plot(time, x[:, 4], label='vy')
plt.plot(time, x[:, 5], label='omega')
plt.title('Velocities')
plt.xlabel('Time [s]')
plt.ylabel('Velocity')
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(time, u[:, 0], label='Thrust F')
plt.plot(time, u[:, 1], label='Torque τ')
plt.title('Optimal Control Inputs')
plt.xlabel('Time [s]')
plt.ylabel('Control')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
