import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

# Constants
m = 0.250           # mass (kg)
g = 9.81          # gravity (m/s^2)
Ixx = 0.01        # moment of inertia (kg.m^2)
T = 50.0           # final time (s)
N = 300           # number of time steps
dt = T / N
time_grid = np.linspace(0, T, N)

# Initial and goal states: [x, y, theta, vx, vy, omega]
x0 = np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0])
x_goal = np.array([6.0, 6.0, 0.0, 0.0, 0.0, 0.0])

# Initialize state and costate
x = np.zeros((N, 6))
p = np.zeros((N, 6))
u = np.zeros((N, 2))  # [F, tau]

x[0] = x0
p[-1] = x[-1] - x_goal  # terminal condition (soft constraint)

# Track cost at each step
cost_values = np.zeros(N)

# Time the simulation
start_time = time.time()

# Forward-backward integration loop (shooting method)
for iteration in range(10):  # 10 iterations of forward-backward sweep
    # Forward: integrate state
    for i in range(N - 1):
        theta = x[i, 2]

        # Optimal controls from costate
        F = (p[i, 3] * np.sin(theta) - p[i, 4] * np.cos(theta)) / m
        tau = -p[i, 5] / Ixx
        u[i] = [F, tau]

        # State dynamics
        dx = np.array([
            x[i, 3],
            x[i, 4],
            x[i, 5],
            -F * np.sin(theta) / m,
            F * np.cos(theta) / m - g,
            tau / Ixx
        ])

        x[i + 1] = x[i] + dt * dx

    # Backward: integrate costate
    p[-1] = x[-1] - x_goal
    for i in reversed(range(N - 1)):
        xi = x[i]
        pi = p[i + 1]
        theta = xi[2]
        F = u[i, 0]

        dH_dx = np.array([
            xi[0] - x_goal[0],
            xi[1] - x_goal[1],
            xi[2] - x_goal[2] - (F / m) * (pi[3] * np.cos(theta) + pi[4] * np.sin(theta)),
            xi[3] - x_goal[3] - pi[0],
            xi[4] - x_goal[4] - pi[1],
            xi[5] - x_goal[5] - pi[2]
        ])

        p[i] = pi + dt * dH_dx

    # Update cost function
    for i in range(N):
        state_cost = 0.5 * np.linalg.norm(x[i] - x_goal)**2
        control_cost = 0.5 * np.linalg.norm(u[i])**2
        cost_values[i] = state_cost + control_cost

end_time = time.time()
final_cost = cost_values.sum() * dt
compute_time = end_time - start_time


# Trajectory plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x[:, 0], x[:, 1], label="Trajectory")
plt.plot(x0[0], x0[1], 'go', label="Start")
plt.plot(x_goal[0], x_goal[1], 'ro', label="Goal")
plt.title("2D Quadrotor Trajectory")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid()
plt.legend()

# Costate evolution
plt.subplot(1, 2, 2)
for i in range(6):
    plt.plot(time_grid, p[:, i], label=f'p{i+1}')
plt.title("Costate Evolution")
plt.xlabel("Time (s)")
plt.ylabel("Costate Values")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Cost plot
plt.figure()
plt.plot(time_grid, cost_values)
plt.title("Instantaneous Cost")
plt.xlabel("Time (s)")
plt.ylabel("Cost")
plt.grid()
plt.show()


print("----- SIMULATION RESULTS -----")
print(f"Final Cost (J): {final_cost:.4f}")
print(f"Total Computation Time: {compute_time:.4f} s")
print(f"Final State:\n{x[-1]}")
print(f"Goal State:\n{x_goal}")
print("\nCostate Values at Final Time:")
print(p[-1])
