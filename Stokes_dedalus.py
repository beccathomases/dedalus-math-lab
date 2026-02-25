import numpy as np
import dedalus.public as d3

# Parameters
Lx = 2*np.pi
Ly = 2*np.pi
Nx = Ny = 128
beta = 1.0
dealias = 3/2

coords = d3.CartesianCoordinates('x','y')
dist = d3.Distributor(coords, dtype=np.float64)

xb = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
yb = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)

x, y = dist.local_grids(xb, yb)
ex, ey = coords.unit_vector_fields(dist)

# Unknowns
u  = dist.VectorField(coords, name='u', bases=(xb, yb))
p  = dist.Field(name='p', bases=(xb, yb))

# Gauge scalars (no bases => constants)
ux0 = dist.Field(name='ux0')
uy0 = dist.Field(name='uy0')
p0  = dist.Field(name='p0')

# Forcing
f = dist.VectorField(coords, name='f', bases=(xb, yb))
f['g'][0] =  2*np.sin(x)*np.cos(y)
f['g'][1] = -2*np.cos(x)*np.sin(y)

problem = d3.LBVP([u, p, ux0, uy0, p0], namespace=locals())

# Stokes + gauges
problem.add_equation("beta*div(grad(u)) - grad(p) + ux0*ex + uy0*ey = -f")
problem.add_equation("div(u) + p0 = 0")

# Fix means (removes nullspaces)
problem.add_equation("integ(p) = 0")
problem.add_equation("integ(u@ex) = 0")
problem.add_equation("integ(u@ey) = 0")

solver = problem.build_solver()
solver.solve()

# Analytic check: for this forcing, p=0 and u = f/(2*beta)
u.change_scales(1)
f.change_scales(1)
print("max|u - f/(2beta)|:", np.max(np.abs(u['g'] - f['g']/(2*beta))))
