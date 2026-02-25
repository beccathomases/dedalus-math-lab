import numpy as np
import dedalus.public as d3
from mpi4py import MPI

# -------------------------
# Parameters
# -------------------------
Lx = 2*np.pi
Ly = 2*np.pi
Nx = Ny = 128
beta = 1.0
dealias = 3/2

A_couple = 0.5     # coupling strength in forcing: +A*(q-mean(q))*ex
kappa_q  = 1e-2    # scalar diffusion
dt_step       = 5e-4
t_end    = 0.5

comm = MPI.COMM_WORLD

def global_sum(a):
    return comm.allreduce(np.sum(a), op=MPI.SUM)

def q2_centroid(q, x, y):
    q.change_scales(1)
    qg = q['g']
    w = qg*qg  # weight = q^2 (always >=0)

    W  = global_sum(w)
    xW = global_sum(w * x)
    yW = global_sum(w * y)

    # avoid divide-by-zero if q is identically 0
    if W == 0:
        return np.nan, np.nan

    return xW/W, yW/W


# -------------------------
# Domain
# -------------------------
coords = d3.CartesianCoordinates('x','y')
dist = d3.Distributor(coords, dtype=np.float64)

xb = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
yb = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)

x, y = dist.local_grids(xb, yb)
ex, ey = coords.unit_vector_fields(dist)

# Derivatives
dx = lambda F: d3.Differentiate(F, coords['x'])
dy = lambda F: d3.Differentiate(F, coords['y'])

grad = d3.grad
div  = d3.div


# -------------------------
# Unknowns for Stokes
# -------------------------
u  = dist.VectorField(coords, name='u', bases=(xb, yb))
p  = dist.Field(name='p', bases=(xb, yb))

# Gauges (constants)
ux0 = dist.Field(name='ux0')
uy0 = dist.Field(name='uy0')
p0  = dist.Field(name='p0')

# Forcing field (will be updated each step)
f = dist.VectorField(coords, name='f', bases=(xb, yb))

# Base "4-roll-mill-like" forcing (your earlier one)
f_roll = dist.VectorField(coords, name='f_roll', bases=(xb, yb))
f_roll['g'][0] =  2*np.sin(x)*np.cos(y)
f_roll['g'][1] = -2*np.cos(x)*np.sin(y)

# -------------------------
# Scalar field q (advected/diffused)
# -------------------------
q = dist.Field(name='q', bases=(xb, yb))

# Mean helper (global mean on periodic box)
def global_mean(field):
    field.change_scales(1)
    local_sum = np.sum(field['g'])
    local_n   = field['g'].size
    gsum = comm.allreduce(local_sum, op=MPI.SUM)
    gn   = comm.allreduce(local_n,   op=MPI.SUM)
    return gsum/gn

def zero_mean_inplace(field):
    m = global_mean(field)
    field.change_scales(1)
    field['g'] -= m

# Initial condition for q (mean-free)
q['g'] = np.exp(-((x-np.pi)**2 + (y-np.pi/2)**2)/(0.3**2))
zero_mean_inplace(q)

# -------------------------
# Stokes LBVP
# -------------------------
stokes = d3.LBVP([u, p, ux0, uy0, p0], namespace=locals())
stokes.add_equation("beta*div(grad(u)) - grad(p) + ux0*ex + uy0*ey = -f")
stokes.add_equation("div(u) + p0 = 0")
stokes.add_equation("integ(p) = 0")
stokes.add_equation("integ(u@ex) = 0")
stokes.add_equation("integ(u@ey) = 0")

stokes_solver = stokes.build_solver()

# -------------------------
# Scalar IVP: q_t + u·∇q = kappa Δq
# (u is treated as a known coefficient field and updated each step)
# -------------------------
qprob = d3.IVP([q], namespace=locals())
qprob.add_equation("dt(q) - kappa_q*div(grad(q)) = -(u@grad(q))")
qsolver = qprob.build_solver(d3.RK443)
qsolver.stop_sim_time = t_end

def mode_phase_q(q, mx=1, my=1):
    q.change_scales(1)
    q.require_coeff_space()
    # coeff array is local; easiest is to reconstruct with global sums of the complex coefficient
    # Dedalus stores real-fourier coeffs; simplest robust diagnostic: project in grid space instead
    # using inner products against cos/sin.
    xw = np.cos(mx*x) * np.cos(my*y)
    yw = np.sin(mx*x) * np.cos(my*y)

    a_loc = np.sum(q['g'] * xw)
    b_loc = np.sum(q['g'] * yw)

    a = comm.allreduce(a_loc, op=MPI.SUM)
    b = comm.allreduce(b_loc, op=MPI.SUM)

    phase = np.arctan2(b, a)
    amp   = np.sqrt(a*a + b*b)
    return amp, phase


# -------------------------
# Forcing update: f = f_roll + A*(q-mean(q))*ex
# IMPORTANT: (q - mean(q)) ensures mean(f)=0 for periodic Stokes solvability
# -------------------------
def update_forcing_from_q():
    # Ensure we're on the same grid scale
    q.change_scales(1)
    f.change_scales(1)
    f_roll.change_scales(1)

    # mean-free q
    qm = global_mean(q)
    qmf = q['g'] - qm

    f['g'][0] = f_roll['g'][0] + A_couple * qmf
    f['g'][1] = f_roll['g'][1]

# -------------------------
# One-line analytic check at t=0 for the roll forcing alone:
# if A_couple=0, solution should satisfy u = f_roll/(2*beta), p=0.
# We'll temporarily check that, then restore coupled forcing.
# -------------------------
update_forcing_from_q()
if abs(A_couple) < 1e-14:
    stokes_solver.solve()
    u.change_scales(1); f_roll.change_scales(1)
    print("max|u - f_roll/(2beta)|:", np.max(np.abs(u['g'] - f_roll['g']/(2*beta))))
else:
    # still useful sanity: check the uncoupled analytic relation once
    # (temporarily set coupling to zero)
    f.save = f['g'].copy()
    f.change_scales(1)
    f['g'][0] = f_roll['g'][0]
    f['g'][1] = f_roll['g'][1]
    stokes_solver.solve()
    u.change_scales(1); f_roll.change_scales(1)
    print("uncoupled check max|u - f_roll/(2beta)|:", np.max(np.abs(u['g'] - f_roll['g']/(2*beta))))
    # restore will happen next update

def L2_sq(field):
    field.change_scales(1)
    g = field['g']
    loc = np.sum(g*g)
    return comm.allreduce(loc, op=MPI.SUM) / comm.allreduce(g.size, op=MPI.SUM)

def grad_L2_sq(field):
    field.change_scales(1)
    fx = dx(field).evaluate(); fx.change_scales(1)
    fy = dy(field).evaluate(); fy.change_scales(1)
    gx, gy = fx['g'], fy['g']
    loc = np.sum(gx*gx + gy*gy)
    return comm.allreduce(loc, op=MPI.SUM) / comm.allreduce(gx.size, op=MPI.SUM)


# -------------------------
# Time loop (explicit coupling per step)
#   Validation mode: FREEZE_U=True => solve Stokes once and only advect/diffuse q
# -------------------------
FREEZE_U = False   # <-- set True for scalar validation; set False to re-enable coupling
A_couple = 0.05   # start 10x smaller than 0.5

dx_phys = Lx / Nx
dy_phys = Ly / Ny

def global_int_array(a):
    # integral over domain of array a (assumes a is at scale=1 grid)
    return comm.allreduce(np.sum(a), op=MPI.SUM) * dx_phys * dy_phys

def scalar_L2(q):
    q.change_scales(1)
    return np.sqrt(global_int_array(q['g']**2))

def cfl_number(u):
    u.change_scales(1)
    umax = comm.allreduce(np.max(np.abs(u['g'][0])), op=MPI.MAX)
    vmax = comm.allreduce(np.max(np.abs(u['g'][1])), op=MPI.MAX)
    return dt_step * max(umax/dx_phys, vmax/dy_phys)

# --- If freezing u, build forcing once and solve Stokes once ---
if FREEZE_U:
    # Use roll forcing only (decoupled) to validate advection/diffusion
    f.change_scales(1); f_roll.change_scales(1)
    f['g'][0] = f_roll['g'][0]
    f['g'][1] = f_roll['g'][1]
    stokes_solver.solve()

t = 0.0
it = 0

L2_0 = scalar_L2(q)

while t < t_end - 1e-14:

    if not FREEZE_U:
        # 1) build forcing from current q
        update_forcing_from_q()

        # 2) solve Stokes for u,p with that forcing
        stokes_solver.solve()

    # 3) advance q one step using this u
    qsolver.step(dt_step)

    # keep q mean-free (prevents creating a constant forcing mode)
    zero_mean_inplace(q)

    t += dt_step
    it += 1

    if it % 50 == 0:
        u.change_scales(1)
        q.change_scales(1)

        speed = np.sqrt(u['g'][0]**2 + u['g'][1]**2)
        umax = comm.allreduce(np.max(speed), op=MPI.MAX)
        qmax = comm.allreduce(np.max(np.abs(q['g'])), op=MPI.MAX)

        qmean = global_mean(q)
        L2 = scalar_L2(q)
        CFL = cfl_number(u)

        # centroid of q^2 (global, robust)
        xc, yc = q2_centroid(q, x, y)

        if comm.rank == 0:
            tag = "FREEZE_U" if FREEZE_U else "COUPLED"
            print(f"[{tag}] it={it:5d} t={t:.4f}  CFL={CFL:.3e}  max|u|={umax:.3e}  "
                  f"max|q|={qmax:.3e}  mean(q)={qmean:.3e}  L2(q)/L2(0)={L2/L2_0:.6f}  "
                  f"(x_c,y_c)=({xc:.3f},{yc:.3f})")
        # divergence diagnostic
        divu = div(u).evaluate()
        divu.change_scales(1)
        div_inf = comm.allreduce(np.max(np.abs(divu['g'])), op=MPI.MAX)
        if comm.rank == 0:
            print(f"    ||div u||_inf = {div_inf:.3e}")

        L2q2   = L2_sq(q)
        G2     = grad_L2_sq(q)

        # expected d/dt (1/2||q||^2) = -kappa * ||grad q||^2  (for incompressible u, periodic)
        rhs = -kappa_q * G2

        # finite-difference estimate of d/dt (1/2||q||^2)
        if it == 50:
            prev_L2q2 = L2q2
        else:
            ddt = 0.5*(L2q2 - prev_L2q2) / (50*dt_step)
            if comm.rank == 0:
                print(f"    energy: d/dt(1/2||q||^2)={ddt:.3e}   -kappa||∇q||^2={rhs:.3e}   defect={(ddt-rhs):.3e}")
            prev_L2q2 = L2q2


