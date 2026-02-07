import numpy as np
import dedalus.public as d3
import logging
from mpi4py import MPI


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ========================
# Parameters
# ========================
Lx, Lz = 4, 1
Nx, Nz = 256, 64
Reynolds = 1
Schmidt = 1
dealias = 3/2
stop_sim_time = 5
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
# Spectral resolution monitors
# ========================
comm = getattr(dist, "comm_cart", dist.comm)  # works in MPI + serial

def spectral_tails(field, frac=0.10):
    """
    Return (tail_x, tail_z, spec_x, spec_z) where tails are energy fractions in
    the highest frac of modes in x and z, computed from coefficient-space energy.
    """
    field.change_scales(1)          # ensure we're at base scales
    c = field['c']                  # coefficient array (local chunk in MPI)
    E = np.abs(c)**2

    # local totals
    local_Etot = np.sum(E)

    # pick how many modes in the "tail"
    nx_loc, nz_loc = E.shape
    kx_tail = max(1, int(frac * Nx))
    kz_tail = max(1, int(frac * Nz))

    # energy by x-mode and z-mode (local)
    local_spec_x = np.sum(E, axis=1)   # sum over z -> energy per x coefficient (local)
    local_spec_z = np.sum(E, axis=0)   # sum over x -> energy per z coefficient (local)

    # tail energies (local)
    local_tail_x = np.sum(E[-kx_tail:, :])   # highest x indices
    local_tail_z = np.sum(E[:, -kz_tail:])   # highest z indices

    # reduce totals
    Etot  = comm.allreduce(local_Etot, op=MPI.SUM)
    tailx = comm.allreduce(local_tail_x, op=MPI.SUM) / (Etot + 1e-300)
    tailz = comm.allreduce(local_tail_z, op=MPI.SUM) / (Etot + 1e-300)

    # gather full spectra by summing across ranks:
    # For x spectrum, ranks own different x slabs, so we allreduce the local arrays
    # into a full-length array only if the local arrays are all length Nx.
    # In Dedalus MPI layouts, local_spec_x is local-length; so we instead just return
    # tail fractions robustly, and optionally return the z spectrum (which is full-length).
    spec_z = comm.allreduce(local_spec_z, op=MPI.SUM)

    return tailx, tailz, spec_z


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
#   (These 4 lines are "sacred": don't change unless you mean to.)
# ========================
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)

grad_u = d3.grad(u) + ez*lift(tau_u1)
grad_s = d3.grad(s) + ez*lift(tau_s1)

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
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

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

        # print basic progress every 10 iters (rank 0 only)
        if comm.rank == 0 and (solver.iteration-1) % 10 == 0:
            logger.info('iter=%i, t=%e, dt=%e', solver.iteration, solver.sim_time, dt)

        # spectral tail monitor every 50 iters
        if (solver.iteration-1) % 50 == 0:
            tailx, tailz, spec_z = spectral_tails(s, frac=0.10)

            if comm.rank == 0:
                spec_z_norm = spec_z / (np.sum(spec_z) + 1e-300)
                last = spec_z_norm[-6:]
                logger.info(
                    "tails (s): x-tail=%.2e, z-tail=%.2e | last Cheb frac=%s",
                    tailx, tailz, np.array2string(last, precision=2, floatmode='fixed')
                )

except Exception:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
