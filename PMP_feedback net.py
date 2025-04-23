import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# loading data
data = torch.load('pmp_data.pt')
x_data = data['x'].detach()
F_data = data['F'].detach()
tau_data = data['tau'].detach()
u_data = torch.stack([F_data, tau_data], dim=1)

device = "cuda" if torch.cuda.is_available() else "cpu"

# input data
x_mean = x_data.mean(0, keepdim=True)
x_std = x_data.std(0, keepdim=True)
x_norm = (x_data - x_mean) / (x_std + 1e-6)

# artificial noise
noise = 0.01 * torch.randn_like(x_norm)
x_aug = x_norm + noise

# network for learning feedback policy
class NNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        out = self.net(x)
        F = torch.clamp(out[:, 0], 0.0, 15.0).unsqueeze(1)    # safe clamp
        tau = torch.clamp(out[:, 1], -5.0, 5.0).unsqueeze(1)
        return torch.cat([F, tau], dim=1)

# init
policy = NNPolicy().to(device)
optimizer = torch.optim.RMSprop(policy.parameters(), lr=1e-3, alpha=0.99, eps=1e-8)

# Train
states_tensor = x_aug.float().to(device)
controls_tensor = u_data.float().to(device)

for epoch in range(1000):
    pred_u = policy(states_tensor)
    loss = ((pred_u - controls_tensor)**2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: RMSProp Loss = {loss.item():.6f}")


def learned_feedback_policy(state):
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32).to(device)
    if state.ndim == 1:
        state = state.unsqueeze(0)

    norm_state = (state - x_mean.to(device)) / (x_std.to(device) + 1e-6)
    with torch.no_grad():
        control = policy(norm_state).squeeze()
    return control[0].item(), control[1].item()

# simulate_policy function
def simulate_policy(policy_fn, x0, T=5.0, N=200):
    dt = T / N
    m = 0.25
    I = 0.0083
    g = 9.81

    traj = [x0.clone()]
    x_t = x0.clone().to(device)

    for _ in range(N - 1):
        F, tau = policy_fn(x_t)
        theta = x_t[2]
        vx, vy, omega = x_t[3], x_t[4], x_t[5]

        dx = torch.tensor([
            vx,
            vy,
            omega,
            -F * torch.sin(theta) / m,
            F * torch.cos(theta) / m - g,
            tau / I
        ], device=device)

        x_t = x_t + dt * dx
        traj.append(x_t.clone())

    return torch.stack(traj)


x0_val = torch.tensor([0.0, 0.0, 0.4, 0.0, 0.0, 0.0], device=device)
rollout = simulate_policy(learned_feedback_policy, x0_val)

rollout_np = rollout.cpu().numpy()
plt.figure(figsize=(10, 4))
plt.plot(rollout_np[:, 0], rollout_np[:, 1], label="Learned Feedback Trajectory")
plt.scatter([x0_val[0].cpu()], [x0_val[1].cpu()], color='green', label="Start")
plt.scatter([5.0], [7.0], color='red', label="Goal")
plt.xlabel("x [m]"); plt.ylabel("y [m]")
plt.title("Trajectory Using Learned Policy (RMSProp)")
plt.legend(); plt.grid()
plt.tight_layout()
plt.show()

#plotting control inputs
with torch.no_grad():
    norm_states = (rollout - x_mean.to(device)) / (x_std.to(device) + 1e-6)
    u_traj = policy(norm_states).cpu().numpy()

plt.figure(figsize=(10, 4))
plt.plot(u_traj[:, 0], label="Thrust F(t)")
plt.plot(u_traj[:, 1], label="Torque τ(t)")
plt.title("Control Inputs from Learned Policy")
plt.xlabel("Time Step")
plt.ylabel("Control")
plt.grid(); plt.legend()
plt.tight_layout()
plt.show()
