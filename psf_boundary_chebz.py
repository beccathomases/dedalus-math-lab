import numpy as np
import dedalus.public as d3
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ========================
# Parameters
# ========================
Lx, Lz = 4, 1
Nx, Nz = 256, 128
Reynolds = 5e3
Schmidt = 1
dealias = 3/2
stop_sim_time = 20
timestepper = d3.RK222
max_timestep = 1e-2
dtype = np.float64
A = 1.0

nu = 1 / Reynolds
D  = nu / Schmidt

# ========================
# Domain / bases
#   Fourier in x, Chebyshev in z
# ========================
coords = d3.CartesianCoordinates('x', 'z')
dist   = d3.Distributor(coords, dtype=dtype)

xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)

x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)

# ========================
# Fields
# ========================
p = dist.Field(name='p', bases=(xbasis, zbasis))
s = dist.Field(name='s', bases=(xbasis, zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))

# Tau fields for first-order tau method
tau_p  = dist.Field(name='tau_p')  # no bases

tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

tau_s1 = dist.Field(name='tau_s1', bases=xbasis)
tau_s2 = dist.Field(name='tau_s2', bases=xbasis)

# ========================
# Lift + first-order reductions
#   (These 3 lines are "sacred": don't change unless you mean to.)
# ========================
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)

grad_u = d3.grad(u) + ez*lift(tau_u1)
grad_s = d3.grad(s) + ez*lift(tau_s1)

ux = u@ex
uz = u@ez
dx = lambda A: d3.Differentiate(A, coords['x'])

dux_dz = (ez @ grad_u) @ ex     # <-- THIS is ∂z u_x
omega  = dx(uz) - dux_dz        # ω = ∂x u_z - ∂z u_x


# ========================
# Problem
# ========================
problem = d3.IVP([p, s, u, tau_p, tau_u1, tau_u2, tau_s1, tau_s2],
                 namespace=locals())

# Incompressibility (with tau_p)
problem.add_equation("trace(grad_u) + tau_p = 0")

# Momentum and tracer (use div(grad_*) instead of lap(*))
problem.add_equation("dt(u) + grad(p) - nu*div(grad_u) + lift(tau_u2) = - u@grad(u)")
problem.add_equation("dt(s) - D*div(grad_s) + lift(tau_s2) = - u@grad(s)")

# No-slip walls
problem.add_equation(f"u(z={-Lz/2}) = 0")
problem.add_equation(f"u(z={+Lz/2}) = 0")

# Tracer BCs (Dirichlet baseline)
problem.add_equation(f"s(z={-Lz/2}) = 0")
problem.add_equation(f"s(z={+Lz/2}) = 0")

# Pressure gauge
problem.add_equation("integ(p) = 0")

# ========================
# Solver
# ========================
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# ========================
# Initial conditions
# ========================
# Shear profile: zero at walls
u['g'][0] = A * np.sin(np.pi * (z + Lz/2) / Lz)
u['g'][1] = 0.0

eps = 0.05
u['g'][1] += eps * np.sin(2*np.pi*x/Lx) * np.sin(np.pi*(z + Lz/2)/Lz)


# Tracer IC compatible with s=0 at walls
s['g'] = u['g'][0]

# Interior perturbations to u_z
u['g'][1] += 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z-0.25)**2/0.01)
u['g'][1] += 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z+0.25)**2/0.01)

# ========================
# Analysis / output
# ========================
snapshots = solver.evaluator.add_file_handler('snapshots-channel', sim_dt=0.1, max_writes=50)
snapshots.add_task(s, name='tracer')
snapshots.add_task(p, name='pressure')
snapshots.add_task(u@ex, name='u_x')
snapshots.add_task(u@ez, name='u_z')
#snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
snapshots.add_task(omega, name='vorticity')


# ========================
# CFL
# ========================
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# ========================
# Main loop
# ========================
try:
    logger.info('Starting main loop')
    while solver.proceed:
        dt = CFL.compute_timestep()
        solver.step(dt)

        if (solver.iteration-1) % 10 == 0:
            logger.info('iter=%i, t=%e, dt=%e', solver.iteration, solver.sim_time, dt)

except Exception:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
