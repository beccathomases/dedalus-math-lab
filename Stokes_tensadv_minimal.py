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

# Linear conformation stepping-stone params
kappa_c = 1e-2      # diffusion on C components
tauR    = 1.0       # relaxation time (C -> I)
dt_step = 5e-4
t_end   = 0.5

comm = MPI.COMM_WORLD

dx_phys = Lx / Nx
dy_phys = Ly / Ny

def global_int_array(a):
    # integral over domain of array a (assumes a is at scale=1 grid)
    return comm.allreduce(np.sum(a), op=MPI.SUM) * dx_phys * dy_phys

# -------------------------
# Domain
# -------------------------
coords = d3.CartesianCoordinates('x','y')
dist = d3.Distributor(coords, dtype=np.float64)

xb = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
yb = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)

x, y = dist.local_grids(xb, yb)
ex, ey = coords.unit_vector_fields(dist)

dx = lambda F: d3.Differentiate(F, coords['x'])
dy = lambda F: d3.Differentiate(F, coords['y'])
grad = d3.grad
div  = d3.div

def cfl_number(u):
    u.change_scales(1)
    umax = comm.allreduce(np.max(np.abs(u['g'][0])), op=MPI.MAX)
    vmax = comm.allreduce(np.max(np.abs(u['g'][1])), op=MPI.MAX)
    return dt_step * max(umax/dx_phys, vmax/dy_phys)

# -------------------------
# Unknowns for Stokes
# -------------------------
u  = dist.VectorField(coords, name='u', bases=(xb, yb))
p  = dist.Field(name='p', bases=(xb, yb))

ux0 = dist.Field(name='ux0')  # gauges (constants)
uy0 = dist.Field(name='uy0')
p0  = dist.Field(name='p0')

# Forcing: 4-roll-mill
f = dist.VectorField(coords, name='f', bases=(xb, yb))
f['g'][0] =  2*np.sin(x)*np.cos(y)
f['g'][1] = -2*np.cos(x)*np.sin(y)

# -------------------------
# Conformation tensor components (2D symmetric)
# -------------------------
cxx = dist.Field(name='cxx', bases=(xb, yb))
cxy = dist.Field(name='cxy', bases=(xb, yb))
cyy = dist.Field(name='cyy', bases=(xb, yb))

# Initial condition: identity + blob
blob = np.exp(-((x-np.pi)**2 + (y-np.pi/2)**2)/(0.3**2))
cxx['g'] = 1.0 + 0.2*blob
cxy['g'] = 0.0
cyy['g'] = 1.0 + 0.2*blob

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
stokes_solver.solve()

# One-line analytic check for this forcing: u = f/(2*beta), p=0
u.change_scales(1); f.change_scales(1)
if comm.rank == 0:
    print("max|u - f/(2beta)|:", np.max(np.abs(u['g'] - f['g']/(2*beta))))

# -------------------------
# Conformation IVP: advection + diffusion + linear relaxation to I
# Ct + u·∇C = kappa ΔC - (1/tauR)(C - I)
# -------------------------
cprob = d3.IVP([cxx, cxy, cyy], namespace=locals())
cprob.add_equation("dt(cxx) - kappa_c*div(grad(cxx)) = -(u@grad(cxx)) - (1/tauR)*(cxx - 1)")
cprob.add_equation("dt(cxy) - kappa_c*div(grad(cxy)) = -(u@grad(cxy)) - (1/tauR)*(cxy)")
cprob.add_equation("dt(cyy) - kappa_c*div(grad(cyy)) = -(u@grad(cyy)) - (1/tauR)*(cyy - 1)")
csolver = cprob.build_solver(d3.RK443)
csolver.stop_sim_time = t_end

# -------------------------
# Diagnostics
# -------------------------
def trC_mean():
    cxx.change_scales(1); cyy.change_scales(1)
    tr = cxx['g'] + cyy['g']
    return global_int_array(tr) / (Lx*Ly)

def trC_L2_sq_int():
    cxx.change_scales(1); cyy.change_scales(1)
    tr = cxx['g'] + cyy['g']
    return global_int_array(tr*tr)

def trC_grad_sq_int():
    # ∫ |∇(cxx+cyy)|^2
    trF = cxx + cyy
    trx = dx(trF).evaluate(); trx.change_scales(1)
    try_ = dy(trF).evaluate(); try_.change_scales(1)
    return global_int_array(trx['g']**2 + try_['g']**2)

# -------------------------
# Time loop
# -------------------------
t = 0.0
it = 0
prev_tr2 = None

while t < t_end - 1e-14:

    csolver.step(dt_step)
    t += dt_step
    it += 1

    if it % 50 == 0:
        # conformation summaries
        cxx.change_scales(1); cxy.change_scales(1); cyy.change_scales(1)
        tr = cxx['g'] + cyy['g']

        tr_mean = trC_mean()
        tr_min  = comm.allreduce(np.min(tr), op=MPI.MIN)
        tr_max  = comm.allreduce(np.max(tr), op=MPI.MAX)

        cxx_min = comm.allreduce(np.min(cxx['g']), op=MPI.MIN)
        cxx_max = comm.allreduce(np.max(cxx['g']), op=MPI.MAX)

        cxy_min = comm.allreduce(np.min(cxy['g']), op=MPI.MIN)
        cxy_max = comm.allreduce(np.max(cxy['g']), op=MPI.MAX)

        # incompressibility diagnostic
        divu = div(u).evaluate(); divu.change_scales(1)
        div_inf = comm.allreduce(np.max(np.abs(divu['g'])), op=MPI.MAX)

        CFL = cfl_number(u)

        if comm.rank == 0:
            print(f"[C-LIN] it={it:5d} t={t:.4f}  CFL={CFL:.3e}  ||div u||_inf={div_inf:.3e}  "
                  f"trC mean={tr_mean:.6f}  trC min/max=({tr_min:.3e},{tr_max:.3e})  "
                  f"cxx min/max=({cxx_min:.3e},{cxx_max:.3e})  "
                  f"cxy min/max=({cxy_min:.3e},{cxy_max:.3e})")


        # --- Energy check on trC: d/dt 1/2 ∫ tr^2 ---
        tr2 = trC_L2_sq_int()
        G2  = trC_grad_sq_int()

        relax_term = -(1/tauR) * global_int_array(tr*(tr-2.0))
        rhs = -kappa_c * G2 + relax_term

        if prev_tr2 is None:
            prev_tr2 = tr2
        else:
            ddt = 0.5*(tr2 - prev_tr2) / (50*dt_step)
            if comm.rank == 0:
                print(f"    tr-energy: d/dt(1/2∫tr^2)={ddt:.3e}   rhs(diff+relax)={rhs:.3e}   defect={(ddt-rhs):.3e}")
            prev_tr2 = tr2
        
