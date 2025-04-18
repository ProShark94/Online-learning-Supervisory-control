import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# === Constants ===
m = 1.0       # mass (kg)
I = 0.01      # moment of inertia (kg·m²)
g = 9.81      # gravity (m/s²)
rho = 0.1     # control penalty
T = 5.0       # total time (s)
N = 100       # number of time steps
dt = T / N    # time step

# === Symbolic Decision Variables ===
X = ca.MX.sym('X', 6, N+1)   # States: [x, y, theta, vx, vy, omega]
U = ca.MX.sym('U', 2, N)     # Controls: [F, tau]

# === Initial and Goal States ===
x0 = np.array([0, 0, 0, 0, 0, 0])
xg = np.array([5, 5, 0, 0, 0, 0])

# === Objective Function ===
cost = 0
for k in range(N):
    state_error = X[:, k] - xg
    control = U[:, k]
    cost += ca.mtimes([state_error.T, state_error]) + rho * ca.mtimes([control.T, control])

# === Dynamics Constraints (Euler integration) ===
constraints = []
for k in range(N):
    xk = X[:, k]
    uk = U[:, k]
    theta = xk[2]
    F = uk[0]
    tau = uk[1]
    
    dx = ca.vertcat(
        xk[3],                               # dx = vx
        xk[4],                               # dy = vy
        xk[5],                               # dtheta = omega
        -(F / m) * ca.sin(theta),           # dvx
        (F / m) * ca.cos(theta) - g,        # dvy
        tau / I                             # domega
    )
    x_next = xk + dt * dx
    constraints.append(X[:, k+1] - x_next)  # enforce dynamics

# === Boundary Conditions ===
constraints.append(X[:, 0] - x0)   # initial state
constraints.append(X[:, -1] - xg)  # final state

# === Stack All Constraints ===
g = ca.vertcat(*constraints)

# === NLP Problem Setup ===
vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
nlp = {'x': vars, 'f': cost, 'g': g}
solver = ca.nlpsol('solver', 'ipopt', nlp)

# === Initial Guess ===
x_init = np.linspace(x0, xg, N+1).T
u_init = np.zeros((2, N))
vars_init = np.concatenate([x_init.flatten(order='F'), u_init.flatten(order='F')])

# === Solve NLP ===
sol = solver(x0=vars_init, lbg=0, ubg=0)
w_opt = sol['x'].full().flatten()

# === Extract Solution ===
X_opt = w_opt[:(6*(N+1))].reshape((6, N+1), order='F')
U_opt = w_opt[(6*(N+1)):].reshape((2, N), order='F')

# === Plot Trajectory ===
plt.figure(figsize=(6, 6))
plt.plot(X_opt[0], X_opt[1], label='Optimal trajectory')
plt.scatter(x0[0], x0[1], color='red', label='Start')
plt.scatter(xg[0], xg[1], color='green', label='Goal')
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("CasADi Optimal Control Trajectory")
plt.grid()
plt.legend()
plt.axis("equal")
plt.show()

# === Optional: Plot Controls ===
t_grid = np.linspace(0, T, N)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t_grid, U_opt[0], label='Thrust F(t)')
plt.ylabel('F [N]')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t_grid, U_opt[1], label='Torque tau(t)', color='orange')
plt.xlabel('Time [s]')
plt.ylabel('Torque [Nm]')
plt.grid()
plt.suptitle("Optimal Control Inputs")
plt.tight_layout()
plt.show()
