import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# Constants
m = 1.0        # mass
I = 0.1        # moment of inertia
g = 9.81       # gravity
R = 1e-4       # control penalty
xd, yd, thetad = 2.0, 2.0, 0.0  # target position and angle

# Time grid
t = np.linspace(0, 5, 200)

# Initial guess for the solution: 12 states (6 state vars + 6 costates)
def initial_guess(t):
    y = np.zeros((12, t.size))
    y[0] = xd * t / t[-1]             # x from 0 to xd
    y[1] = yd * t / t[-1]             # y from 0 to yd
    y[2] = 0.5 * (1 - t / t[-1])      # theta from 0.5 to 0
    y[3] = xd / t[-1]                 # vx roughly constant
    y[4] = yd / t[-1]                 # vy roughly constant
    y[5] = -0.5 / t[-1]               # omega
    y[6:] = 0.1                       # costates nonzero
    return y

# Dynamics + costates (PMP)
def dynamics(t, y):
    x, y_pos, theta, vx, vy, omega = y[0], y[1], y[2], y[3], y[4], y[5]
    l1, l2, l3, l4, l5, l6 = y[6], y[7], y[8], y[9], y[10], y[11]

    # Optimal control laws
    # F = (l4 * np.sin(theta) - l5 * np.cos(theta)) / (2 * R * m)
    # tau = -l6 / (2 * R * I)
    F = np.clip((l4 * np.sin(theta) - l5 * np.cos(theta)) / (2 * R * m), -50, 50)
    tau = np.clip(-l6 / (2 * R * I), -50, 50)


    dydt = np.zeros_like(y)
    # State equations
    dydt[0] = vx
    dydt[1] = vy
    dydt[2] = omega
    dydt[3] = -F * np.sin(theta) / m
    dydt[4] = F * np.cos(theta) / m - g
    dydt[5] = tau / I

    # Costate equations
    dydt[6] = -2 * (x - xd)
    dydt[7] = -2 * (y_pos - yd)
    dydt[8] = (-2 * (theta - thetad)
               - l4 * F * np.cos(theta) / m
               - l5 * F * np.sin(theta) / m)
    dydt[9] = -l1
    dydt[10] = -l2
    dydt[11] = -l3

    return dydt

# Boundary conditions: initial for states, final for costates
def bc(ya, yb):
    bc = np.zeros(12)
    # Initial state
    bc[0] = ya[0]  # x0 = 0
    bc[1] = ya[1]  # y0 = 0
    bc[2] = ya[2] - 0.5  # theta0 = 0.5
    bc[3] = ya[3]        # vx0 = 0
    bc[4] = ya[4]        # vy0 = 0
    bc[5] = ya[5]        # omega0 = 0

    # Terminal costates = 0
    bc[6:] = yb[6:]
    return bc

# Solve BVP
sol = solve_bvp(dynamics, bc, t, initial_guess(t), max_nodes=100000, tol=1e-3, verbose=2)


# Check solution success
if not sol.success:
    print("BVP solver failed:", sol.message)
else:
    print("Solution successful!")

# Extract solution
T = sol.x
S = sol.y

# Recover control inputs
theta = S[2]
l4 = S[9]
l5 = S[10]
l6 = S[11]
F = (l4 * np.sin(theta) - l5 * np.cos(theta)) / (2 * R * m)
tau = -l6 / (2 * R * I)

# Plot state trajectories
plt.figure(figsize=(10, 5))
plt.plot(T, S[0], label="x(t)")
plt.plot(T, S[1], label="y(t)")
plt.plot(T, S[2], label="theta(t)")
plt.title("State Trajectories")
plt.xlabel("Time [s]")
plt.legend()
plt.grid()

# Plot control inputs
plt.figure(figsize=(10, 5))
plt.plot(T, F, label="Thrust F(t)")
plt.plot(T, tau, label="Torque τ(t)")
plt.title("Control Inputs")
plt.xlabel("Time [s]")
plt.legend()
plt.grid()

plt.show()
