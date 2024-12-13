import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
dx = 0.1  # Spatial step
dt = 0.005  # Time step
alpha = 0.01  # Thermal diffusivity
L = 1.0  # Length of the domain
x = np.arange(0, L + dx, dx)  # Spatial grid
nx = len(x)  # Number of grid points
nt = 100  # Number of time steps

# Stability condition
assert dt <= dx**2 / (2 * alpha), "Stability condition not satisfied!"

# Initial and boundary conditions
u = np.sin(np.pi * x)  # Initial condition
time_steps = [u.copy()]  # Store solutions at each time step

# Boundary conditions
u[0] = 0
u[-1] = 0

# Time stepping
for n in range(nt):
    u_new = u.copy()
    for i in range(1, nx - 1):
        u_new[i] = u[i] + (alpha * dt / dx**2) * (u[i + 1] - 2 * u[i] + u[i - 1])
    u = u_new
    time_steps.append(u.copy())

# Animation
fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot(x, time_steps[0], label="Temperature")
ax.set_xlim(0, L)
ax.set_ylim(0, 1)
ax.set_xlabel("x")
ax.set_ylabel("Temperature u(x, t)")
ax.set_title("1D Heat Equation Solution (Animation)")
ax.legend()
ax.grid()

def update(frame):
    line.set_ydata(time_steps[frame])
    ax.set_title(f"1D Heat Equation Solution (t = {frame * dt:.3f})")
    return line,

ani = FuncAnimation(fig, update, frames=len(time_steps), interval=50, blit=True)
plt.show()
