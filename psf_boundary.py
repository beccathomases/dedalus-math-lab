import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

# -----------------------------
# 1. Domain parameters
# -----------------------------
Lx, Lz = 4.0, 1.0        # domain sizes
Nx, Nz = 128, 64         # number of points
A = 1.0                  # amplitude of shear
z0 = Lz / 2              # center of tanh gradient
delta = 0.05             # width of shear layer
dtype = np.float64

# -----------------------------
# 2. Bases and distributor
# -----------------------------
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)

# Fourier in x (periodic), Chebyshev in z (allows steep gradients)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx))
zbasis = d3.Chebyshev(coords['z'], size=Nz, bounds=(0, Lz))

# -----------------------------
# 3. Fields
# -----------------------------
u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))

# -----------------------------
# 4. Initialize background shear
# -----------------------------
x, z = dist.local_grids(xbasis, zbasis)

# Option 1: sine wave (periodic in z, mimics Poisson height BC)
u['g'][0] = A * np.sin(2*np.pi*z / Lz)

# Option 2: tanh layer (soft edges, like virtual walls)
# u['g'][0] = A * np.tanh((z - z0)/delta)   # x-component of velocity
u['g'][1] = 0                             # z-component (vertical)

# -----------------------------
# 5. Plot initial profile to check
# -----------------------------
plt.figure(figsize=(6,4))
plt.plot(z, u['g'][0], label="u_x(z) initial shear")
plt.xlabel("z")
plt.ylabel("Velocity u_x")
plt.title("Initial Background Shear with Soft Boundary")
plt.grid(True)
plt.legend()
plt.show()

# -----------------------------
# 6. (Optional) Add damping near edges to mimic stronger wall
# -----------------------------
kappa = 10.0
damping = dist.Field(bases=(zbasis,))
damping['g'] = -kappa * u['g'][0]  # slows down edges
# This can be added to the RHS of your solver to enforce soft boundary

# -----------------------------
# 7. Setup IVP (if you want to time-evolve)
# -----------------------------
from dedalus.extras import flow_tools

problem = d3.IVP([u], namespace=locals())
nu = 1e-3
problem.add_equation("dt(u) - nu*lap(u) = 0")  # simple diffusion example

solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = 1.0

CFL = d3.CFL(solver, initial_dt=1e-3, cadence=1, safety=0.5,
             max_dt=1e-2, threshold=0.1)
CFL.add_velocity(u)

# -----------------------------
# 8. Run solver and plot evolution
# -----------------------------
while solver.proceed:
    dt = CFL.compute_timestep()
    solver.step(dt)

plt.figure(figsize=(6,4))
plt.plot(z, u['g'][0], label="u_x(z) final shear")
plt.xlabel("z")
plt.ylabel("Velocity u_x")
plt.title("Final Background Shear")
plt.grid(True)
plt.legend()
plt.show()
