'''
This script simulates a 2D quadrotor trajectory using a PD controller.
The quadrotor is modeled with a simple set of dynamics and a controller that computes the thrust and torque required to reach a specified goal position.
The simulation uses the `scipy.integrate.solve_ivp` function to solve the system of ordinary differential equations (ODEs) that describe the quadrotor's motion.'''

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
#m = 1.0
m = 0.250 #kg , for tello or crazyflie
#I = 0.01
I = 0.00083 #kg*m^2, for tello or crazyflie
g = 9.81

# desired pos
xg, yg = 5.0, 5.0

# Controller
def controller(state):
    x, y, theta, vx, vy, omega = state
    
    # Position tracking gains, can bump this up for smoother flow
    Kp_pos = 4.0
    Kd_pos = 4.0
    
    ax_des = Kp_pos * (xg - x) + Kd_pos * (0 - vx)
    ay_des = Kp_pos * (yg - y) + Kd_pos * (0 - vy)
    
    # pitch angle final
    theta_des = np.arctan2(-ax_des, ay_des + g)

    # PD control for orientation
    Kp_theta = 15.0
    Kd_theta = 3.0
    tau = -Kp_theta * (theta - theta_des) - Kd_theta * omega

    # Thrust magnitude
    F = m * np.sqrt(ax_des**2 + (ay_des + g)**2)

    return F, tau

# Dynamics function
def quadrotor_dynamics(t, s):
    F, tau = controller(s)
    x, y, theta, vx, vy, omega = s
    dxdt = vx
    dydt = vy
    dthetadt = omega
    dvxdt = -(F / m) * np.sin(theta)
    dvydt = (F / m) * np.cos(theta) - g
    domegadt = tau / I
    return [dxdt, dydt, dthetadt, dvxdt, dvydt, domegadt]

# Initial state at origin with slight tilt
s0 = [0, 0, 0.1, 0, 0, 0]

# Time setup
T = 10.0
t_eval = np.linspace(0, T, 1000)

# Sim
sol = solve_ivp(quadrotor_dynamics, [0, T], s0, t_eval=t_eval)

# Extract states
x, y = sol.y[0], sol.y[1]


plt.figure(figsize=(6,6))
plt.plot(x, y, label='Trajectory')
plt.scatter(x[0], y[0], color='red', label='Start')
plt.scatter(x[-1], y[-1], color='green', label='End')
plt.plot(xg, yg, 'kx', label='Goal', markersize=10)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("2D Quadrotor Tracjectory")
plt.grid()
plt.axis("equal")
plt.legend()
plt.show()   