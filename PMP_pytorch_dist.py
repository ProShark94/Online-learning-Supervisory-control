import torch
import matplotlib.pyplot as plt

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
T = 5.0
N = 200
dt = T / N
time = torch.linspace(0, T, N).to(device)
m, I, g, R = 1.0, 0.1, 9.81, 1e-4
F_min, F_max = 0.0, 15.0
tau_min, tau_max = -5.0, 5.0

# States
x0_val = torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.0], device=device)
goal_state = torch.tensor([5.0, 5.0, 0.0, 0.0, 0.0, 0.0], device=device)

# Warped Initial Guess
t_frac = torch.linspace(0, 1, N).to(device).unsqueeze(1)
arc = 0.8 * torch.sin(torch.pi * t_frac).squeeze()
x_init = x0_val + t_frac * (goal_state - x0_val)
x_init[:, 1] += arc  # bump y
x = torch.nn.Parameter(x_init.clone().detach())
lam = torch.nn.Parameter(0.01 * torch.randn(N, 6, device=device))

# Wind disturbance
def wind_force(t):
    return 0.5 * torch.sin(2 * torch.pi * t / T)

# Obstacle penalty (smooth barrier)
def compute_combined_obstacle_loss(state):
    x_obs, y_obs, radius = 2.5, 2.5, 0.5
    d2 = (state[:, 0] - x_obs)**2 + (state[:, 1] - y_obs)**2
    barrier = torch.exp(-((d2 - radius**2) / 0.01).clamp(min=-5, max=5))
    return barrier.sum()

# PMP control law
def compute_controls(state, costate):
    theta = state[:, 2]
    l4, l5, l6 = costate[:, 3], costate[:, 4], costate[:, 5]
    F = (l4 * torch.sin(theta) - l5 * torch.cos(theta)) / (2 * R * m)
    tau = -l6 / (2 * R * I)
    return torch.clamp(F, F_min, F_max), torch.clamp(tau, tau_min, tau_max)

# Dynamics
def compute_dynamics(state, control):
    F, tau = control
    F = F[:-1]
    tau = tau[:-1]
    theta = state[:-1, 2]
    vx = state[:-1, 3]
    vy = state[:-1, 4]
    omega = state[:-1, 5]
    wind = wind_force(time[:-1])
    dx = torch.stack([
        vx,
        vy,
        omega,
        -F * torch.sin(theta) / m,
        F * torch.cos(theta) / m - g + wind,
        tau / I
    ], dim=1)
    return dx

# Cost function
def compute_cost(_, control):
    F, tau = control
    return dt * R * (F**2 + tau**2).mean()

# Total loss
def compute_loss():
    F, tau = compute_controls(x, lam)
    dx = compute_dynamics(x, (F, tau))
    dyn_res = x[1:] - (x[:-1] + dt * dx)
    dyn_loss = dyn_res.pow(2).mean()

    costate_res = torch.zeros_like(lam)
    costate_res[1:, 0] = lam[1:, 0] - lam[:-1, 0]
    costate_res[1:, 1] = lam[1:, 1] - lam[:-1, 1]
    costate_res[1:, 2] = lam[1:, 2] - lam[:-1, 2]
    costate_res[1:, 3] = lam[1:, 3] - (lam[:-1, 3] + dt * lam[:-1, 0])
    costate_res[1:, 4] = lam[1:, 4] - (lam[:-1, 4] + dt * lam[:-1, 1])
    costate_res[1:, 5] = lam[1:, 5] - (lam[:-1, 5] + dt * lam[:-1, 2])
    lam_loss = costate_res.pow(2).mean()

    x_start_loss = (x[0] - x0_val).pow(2).mean()
    x_end_loss = (x[-1] - goal_state).pow(2).mean()
    lam_end_loss = lam[-1].pow(2).mean()

    obs_loss = compute_combined_obstacle_loss(x)
    cost = compute_cost(x, (F, tau))

    total_loss = (
        cost +
        100 * dyn_loss +
        100 * lam_loss +
        1000 * x_start_loss +
        1000 * x_end_loss +
        1000 * lam_end_loss +
        10000 * obs_loss
    )
    return total_loss, cost, dyn_loss, lam_loss, obs_loss

# Optimizer
optimizer = torch.optim.LBFGS([x, lam], max_iter=500, tolerance_grad=1e-6, line_search_fn='strong_wolfe')
loss_history = []

def closure():
    optimizer.zero_grad()
    loss, cost, dyn, lamres, obs = compute_loss()
    loss.backward()
    loss_history.append(loss.item())
    if len(loss_history) % 10 == 0:
        print(f"Iter {len(loss_history)} | Loss: {loss:.2f} | ObsLoss: {obs:.2f}")
    return loss

print("Optimizing...")
optimizer.step(closure)
print("Optimization complete.")

# Evaluation
with torch.no_grad():
    F_opt, tau_opt = compute_controls(x, lam)
    total_cost, _, dyn_loss, lam_loss, obs_loss = compute_loss()
    print(f"Final Cost: {total_cost:.4f}, Obstacle Loss: {obs_loss:.4f}")
    total_loss = dyn_loss + lam_loss + obs_loss
    print(f"Total Loss: {total_loss:.4f}")

# Plot results
x_np = x.detach().cpu().numpy()
F_np = F_opt.detach().cpu().numpy()
tau_np = tau_opt.detach().cpu().numpy()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x_np[:, 0], x_np[:, 1], label="Trajectory A→B")
plt.scatter([x0_val[0].cpu()], [x0_val[1].cpu()], label="Start", color='green')
plt.scatter([goal_state[0].cpu()], [goal_state[1].cpu()], label="Goal", color='red')
circle = plt.Circle((2.5, 2.5), 0.5, color='gray', alpha=0.3, label="Obstacle")
plt.gca().add_patch(circle)
plt.title("Trajectory with Obstacle")
plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.grid(); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(time.cpu(), F_np, label="Thrust F(t)")
plt.plot(time.cpu(), tau_np, label="Torque τ(t)")
plt.title("Control Inputs")
plt.xlabel("Time [s]"); plt.ylabel("Control")
plt.grid(); plt.legend()
plt.tight_layout()
plt.show()

vx_np = x_np[:, 3]
vy_np = x_np[:, 4]
theta_np = x_np[:, 2]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(time.cpu(), vx_np, label="v_x(t)")
plt.plot(time.cpu(), vy_np, label="v_y(t)")
plt.title("Velocity")
plt.grid(); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(time.cpu(), theta_np, label="θ(t)", color='purple')
plt.title("Orientation")
plt.grid(); plt.legend()
plt.tight_layout()
plt.show()
