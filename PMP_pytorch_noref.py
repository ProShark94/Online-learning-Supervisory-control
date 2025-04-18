import torch
import matplotlib.pyplot as plt

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# parameters
T = 5.0
N = 200
dt = T / N
time = torch.linspace(0, T, N).to(device)

# Quadrotor constants
#m = 1.0
m = 0.25
#I = 0.1
I = 0.0083
g = 9.81
R = 1e-5  # penalty on control effort

# Initial and goal states
x0_val = torch.tensor([0.0, 0.0, 0.4, 0.0, 0.0, 0.0], device=device)
goal_state = torch.tensor([5.0, 7.0, 0.0, 0.0, 0.0, 0.0], device=device)

# Initialize state and costate trajectories
x = torch.nn.Parameter(torch.linspace(0, 1, N).unsqueeze(1) * (goal_state - x0_val).unsqueeze(0) + x0_val)
lam = torch.nn.Parameter(0.01 * torch.randn(N, 6, device=device))

#PMP control law
def compute_controls(state, costate):
    theta = state[:, 2]
    l4, l5, l6 = costate[:, 3], costate[:, 4], costate[:, 5]
    F = (l4 * torch.sin(theta) - l5 * torch.cos(theta)) / (2 * R * m)
    tau = -l6 / (2 * R * I)
    return F, tau

# Dynamics
def compute_dynamics(state, control):
    x, y, theta, vx, vy, omega = state.T
    F, tau = control
    dx = torch.stack([
        vx,
        vy,
        omega,
        -F * torch.sin(theta) / m,
        F * torch.cos(theta) / m - g,
        tau / I
    ], dim=1)
    return dx

# Cost function
def compute_cost(_, control):
    F, tau = control
    effort = R * (F**2 + tau**2)
    return dt * effort.mean()

# Obstacle penalty (smooth barrier)
def compute_loss():
    F, tau = compute_controls(x, lam)
    dx = compute_dynamics(x, (F, tau))

    # Tracking cost
    dyn_res = x[1:] - (x[:-1] + dt * dx[:-1])
    dyn_loss = dyn_res.pow(2).mean()

    # Costate residuals
    costate_res = torch.zeros_like(lam)
    costate_res[1:, 0] = lam[1:, 0] - lam[:-1, 0]
    costate_res[1:, 1] = lam[1:, 1] - lam[:-1, 1]
    costate_res[1:, 2] = lam[1:, 2] - lam[:-1, 2]
    costate_res[1:, 3] = lam[1:, 3] - (lam[:-1, 3] + dt * lam[:-1, 0])
    costate_res[1:, 4] = lam[1:, 4] - (lam[:-1, 4] + dt * lam[:-1, 1])
    costate_res[1:, 5] = lam[1:, 5] - (lam[:-1, 5] + dt * lam[:-1, 2])
    lam_loss = costate_res.pow(2).mean()

    # Boundary losses
    x_start_loss = (x[0] - x0_val).pow(2).mean()
    x_end_loss = (x[-1] - goal_state).pow(2).mean()
    lam_end_loss = lam[-1].pow(2).mean()

    # Total loss
    cost = compute_cost(x, (F, tau))
    total_loss = cost + 100 * dyn_loss + 100 * lam_loss + 1000 * x_start_loss + 1000 * x_end_loss + 1000 * lam_end_loss
    return total_loss, cost, dyn_loss, lam_loss

# optimization
optimizer = torch.optim.LBFGS([x, lam], max_iter=500, tolerance_grad=1e-6, line_search_fn='strong_wolfe')
loss_history = []

def closure():
    optimizer.zero_grad()
    loss, *_ = compute_loss()
    loss.backward()
    loss_history.append(loss.item())
    return loss

print("Optimizing...")
optimizer.step(closure)
print("Optimization complete.")

# eval
with torch.no_grad():
    F_opt, tau_opt = compute_controls(x, lam)
    total_cost, tracking_cost, dyn_loss, lam_loss = compute_loss()
    print(f"Dyn Residual:     {dyn_loss:.4f}")
    print(f"Costate Residual: {lam_loss:.4f}")
    print(f"Tracking Cost:    {tracking_cost:.4f}")
    total_loss= dyn_loss + lam_loss + tracking_cost
    print(f"Final Cost: {total_cost:.4f}, Total Loss: {total_loss:.4f}")


x_np = x.detach().cpu().numpy()
F_np = F_opt.detach().cpu().numpy()
tau_np = tau_opt.detach().cpu().numpy()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x_np[:, 0], x_np[:, 1], label="Optimal Path A → B")
plt.scatter([x0_val[0].cpu()], [x0_val[1].cpu()], label="Start", color='green')
plt.scatter([goal_state[0].cpu()], [goal_state[1].cpu()], label="Goal", color='red')
plt.xlabel("x [m]"); plt.ylabel("y [m]")
plt.title("Optimal Trajectory")
plt.legend(); plt.grid()

plt.subplot(1, 2, 2)
plt.plot(time.cpu(), F_np, label="Thrust F(t)")
plt.plot(time.cpu(), tau_np, label="Torque τ(t)")
plt.xlabel("Time [s]"); plt.ylabel("Control Input")
plt.title("Optimal Control Effort")
plt.legend(); plt.grid()
plt.tight_layout()
plt.show()

# costate trajectories
lam_np = lam.detach().cpu().numpy()
plt.figure(figsize=(10, 6))
for i in range(6):
    plt.plot(time.cpu(), lam_np[:, i], label=f"λ{i+1}(t)")
plt.xlabel("Time [s]")
plt.ylabel("Costate Value")
plt.title("Costate Trajectories λ(t)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


