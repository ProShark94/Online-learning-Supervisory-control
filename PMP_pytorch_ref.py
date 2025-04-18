import torch
import matplotlib.pyplot as plt

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Simulation parameters
T = 5.0
N = 200
dt = T / N
time = torch.linspace(0, T, N).to(device)

# Quadrotor physical constants
m = 1.0
I = 0.1
g = 9.81
R = 1e-5  # smaller R encourages motion

# Desired final state
xd, yd, thetad = 5.0, 5.0, 0.0
x_ref = xd * time / T
y_ref = yd * time / T
theta_ref = thetad + 0 * time

# Initial state
x0_val = torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.0], device=device)

# --- Smart initialization: interpolate straight line trajectory ---
x_init = torch.stack([
    x_ref, y_ref, theta_ref + 0.0,                   # x, y, theta
    torch.gradient(x_ref, spacing=dt)[0],            # vx
    torch.gradient(y_ref, spacing=dt)[0],            # vy
    torch.gradient(theta_ref, spacing=dt)[0]         # omega
], dim=1)
x = torch.nn.Parameter(x_init.clone().detach())
lam = torch.nn.Parameter(0.01 * torch.randn(N, 6, device=device))  # small random costates

# --- PMP Control Law ---
def compute_controls(state, costate):
    theta = state[:, 2]
    l4, l5, l6 = costate[:, 3], costate[:, 4], costate[:, 5]
    F = (l4 * torch.sin(theta) - l5 * torch.cos(theta)) / (2 * R * m)
    tau = -l6 / (2 * R * I)
    return F, tau

# --- System Dynamics ---
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

# --- Cost Function ---
def compute_cost(state, control):
    x_pos, y_pos, theta = state[:, 0], state[:, 1], state[:, 2]
    F, tau = control
    tracking = (x_pos - x_ref)**2 + (y_pos - y_ref)**2 + (theta - theta_ref)**2
    effort = R * (F**2 + tau**2)
    return dt * (tracking + effort).mean()

# --- Total Loss Function ---
def compute_loss():
    F, tau = compute_controls(x, lam)
    dx = compute_dynamics(x, (F, tau))

    # Forward Euler collocation dynamics residual
    dyn_res = x[1:] - (x[:-1] + dt * dx[:-1])
    dyn_loss = dyn_res.pow(2).mean()

    # Costate dynamics (Euler approximation)
    costate_res = torch.zeros_like(lam)
    costate_res[1:, 0] = lam[1:, 0] - (lam[:-1, 0] + 2 * dt * (x[:-1, 0] - x_ref[:-1]))
    costate_res[1:, 1] = lam[1:, 1] - (lam[:-1, 1] + 2 * dt * (x[:-1, 1] - y_ref[:-1]))
    costate_res[1:, 2] = lam[1:, 2] - (lam[:-1, 2] + 2 * dt * (x[:-1, 2] - theta_ref[:-1]))
    costate_res[1:, 3] = lam[1:, 3] - (lam[:-1, 3] + dt * lam[:-1, 0])
    costate_res[1:, 4] = lam[1:, 4] - (lam[:-1, 4] + dt * lam[:-1, 1])
    costate_res[1:, 5] = lam[1:, 5] - (lam[:-1, 5] + dt * lam[:-1, 2])
    lam_loss = costate_res.pow(2).mean()

    # Boundary and terminal loss
    x_start_loss = (x[0] - x0_val).pow(2).mean()
    lam_end_loss = lam[-1].pow(2).mean()
    x_end_loss = ((x[-1, 0] - xd)**2 + (x[-1, 1] - yd)**2 + (x[-1, 2] - thetad)**2)

    # Main cost (PMP objective)
    cost = compute_cost(x, (F, tau))

    # Total loss
    total_loss = cost + 100 * dyn_loss + 100 * lam_loss + 1000 * x_start_loss + 1000 * lam_end_loss + 1000 * x_end_loss
    return total_loss, cost, dyn_loss, lam_loss

# --- Optimization ---
optimizer = torch.optim.LBFGS([x, lam], max_iter=500, tolerance_grad=1e-6, line_search_fn='strong_wolfe')
loss_history = []

def closure():
    optimizer.zero_grad()
    loss, *_ = compute_loss()
    loss.backward()
    loss_history.append(loss.item())
    return loss

print("🔧 Optimizing...")
optimizer.step(closure)
print("Optimization complete.")

# --- Final Evaluation ---
with torch.no_grad():
    F_opt, tau_opt = compute_controls(x, lam)
    total_cost, tracking_cost, dyn_loss, lam_loss = compute_loss()
    print(f"Final Total Cost: {total_cost:.4f}")
    print(f"Tracking Cost:    {tracking_cost:.4f}")
    print(f"Dyn Residual:     {dyn_loss:.4f}")
    print(f"Costate Residual: {lam_loss:.4f}")

# --- Plot Results ---
x_np = x.detach().cpu().numpy()
F_np = F_opt.detach().cpu().numpy()
tau_np = tau_opt.detach().cpu().numpy()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x_np[:, 0], x_np[:, 1], label="Trajectory")
#plt.plot(x_ref.cpu(), y_ref.cpu(), '--', label="Reference Path")
plt.xlabel("x [m]"); plt.ylabel("y [m]")
plt.title("Quadrotor Trajectory")
plt.legend(); plt.grid()

plt.subplot(1, 2, 2)
plt.plot(time.cpu(), F_np, label="Thrust F(t)")
plt.plot(time.cpu(), tau_np, label="Torque τ(t)")
plt.xlabel("Time [s]"); plt.ylabel("Control Inputs")
plt.title("Optimal Control Inputs")
plt.legend(); plt.grid()
plt.tight_layout()
plt.show()

# --- Costate plot (optional) ---
lam_np = lam.detach().cpu().numpy()
plt.figure(figsize=(10, 4))
for i in range(6):
    plt.plot(time.cpu(), lam_np[:, i], label=f"$lambda_{i+1}(t)$")
plt.xlabel("Time [s]"); plt.ylabel("Costates")
plt.title("Costate Trajectories")
plt.legend(); plt.grid()
plt.tight_layout()
plt.show()
