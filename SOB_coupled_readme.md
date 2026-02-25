# 2D (Periodic) Stokes–Oldroyd-B “4-roll mill” test (Dedalus v3)

This script is a **minimal, periodic 2D** Stokes–Oldroyd-B (SOB) style prototype meant for **sanity checks** and for exporting fields to MATLAB. It evolves a **polymer conformation tensor** under an imposed/feedback-modified Stokes flow driven by a 4-roll-mill body force.

> Key idea: at each time step we (1) build the Stokes forcing from the current polymer state, (2) solve a steady Stokes problem for `u,p`, then (3) advance the conformation components `cxx,cxy,cyy` one time step.

---

## PDEs being solved

### Domain and unknowns
- Spatial domain: \((x,y)\in [0,L_x)\times[0,L_y)\), periodic in both directions.
- Velocity and pressure: \(u(x,y,t)\in \mathbb{R}^2\), \(p(x,y,t)\in \mathbb{R}\).
- Symmetric conformation tensor (2D):  
  \[
  C=\begin{pmatrix}c_{xx} & c_{xy}\\ c_{xy} & c_{yy}\end{pmatrix}.
  \]

### Stokes (steady, re-solved each time step)
The Stokes solve is (in the script’s sign convention):
\[
\beta \Delta u - \nabla p = -f_{\text{total}},\qquad \nabla\cdot u = 0.
\]

Equivalently, 
\[
\beta \Delta u - \nabla p + f_{\text{total}} = 0.
\]

#### Forcing decomposition (notation)
- **Base (fixed) 4-roll forcing** \(f_0(x,y)\):
  \[
  f_0(x,y)=
  \begin{pmatrix}
  2\sin x\cos y\\
  -2\cos x\sin y
  \end{pmatrix}.
  \]
- **Polymer stress feedback** via a divergence of a polymer stress tensor \(\tau_p\):
  \[
  \tau_p = \alpha_p (C - I),
  \qquad
  \nabla\cdot\tau_p =
  \begin{pmatrix}
  \partial_x \tau_{xx} + \partial_y \tau_{xy}\\
  \partial_x \tau_{xy} + \partial_y \tau_{yy}
  \end{pmatrix}.
  \]
- **Total forcing used in Stokes each time step**:
  \[
  f_{\text{total}} = f_0 + \nabla\cdot\tau_p
  = f_0 + \alpha_p \,\nabla\cdot(C-I).
  \]

So the Stokes solve is exactly:
\[
\boxed{\beta \Delta u - \nabla p = -(f_0 + \nabla\cdot\tau_p)}
\]
or equivalently
\[
\boxed{\beta \Delta u - \nabla p + f_0 + \nabla\cdot\tau_p = 0.}
\]

**Important implementation detail:** `update_forcing_from_C()` is called **inside the time loop**, *before* `stokes_solver.solve()`, so \(f_{\text{total}}\) uses the **current** \(C\) and the resulting \(u\) is fed into the conformation step.

---

### Conformation evolution (upper-convected + diffusion + linear relaxation)
We evolve \(C\) by an **upper-convected Oldroyd-B–style** equation with added diffusion and relaxation to the identity:

$\partial_t C + u\cdot\nabla C= (\nabla u)\,C + C(\nabla u)^T+ \kappa_c \Delta C- \frac{1}{\tau_R}(C-I).$


Component form as implemented:

1) \(c_{xx}\):

$\partial_t c_{xx} + u\cdot\nabla c_{xx}= 2(u_{x,x}c_{xx} + u_{x,y}c_{xy})+ \kappa_c \Delta c_{xx}-\frac{1}{\tau_R}(c_{xx}-1).$

2) \(c_{xy}\):

$\partial_t c_{xy} + u\cdot\nabla c_{xy}= u_{x,x}c_{xy} + u_{x,y}c_{yy} + c_{xx}u_{y,x} + c_{xy}u_{y,y}+ \kappa_c \Delta c_{xy}-\frac{1}{\tau_R}c_{xy}.$

3) \(c_{yy}\):

$\partial_t c_{yy} + u\cdot\nabla c_{yy}= 2(u_{y,x}c_{xy} + u_{y,y}c_{yy})+ \kappa_c \Delta c_{yy}-\frac{1}{\tau_R}(c_{yy}-1).$


---

## Time stepping algorithm (what actually happens in the code)

At each step \(n\to n+1\):

1. **Build polymer stress contribution from current conformation**
   - Compute \(\tau_p^n = \alpha_p(C^n-I)\)
   - Compute \((\nabla\cdot\tau_p)^n\)
   - Set \(f_{\text{total}}^n = f_0 + (\nabla\cdot\tau_p)^n\)

2. **Solve steady Stokes**
   \[
   \beta \Delta u^n - \nabla p^n = -f_{\text{total}}^n,\qquad \nabla\cdot u^n=0.
   \]

3. **Advance conformation (RK443)**
   \[
   C^{n+1} \approx C^n + \Delta t \, \mathcal{RHS}(u^n, C^n),
   \]
   where `Dedalus RK443` is used for time integration.

---

## Parameters and defaults (as in the script)

### Geometry / discretization
- `Lx = 2*np.pi`, `Ly = 2*np.pi`  
  Domain size \(L_x=L_y=2\pi\).
- `Nx = Ny = 128`  
  Grid resolution (RealFourier in both directions).
- `dealias = 3/2`  
  3/2 dealiasing factor for nonlinear products.

### Fluid / Stokes
- `beta = 1.0`  
  Viscous coefficient in the Stokes operator \(\beta\Delta u\).  
  (In a nondimensional SOB setting, think of this as the “solvent viscosity factor” used in this prototype.)

### Polymer feedback coupling (Stokes forcing)
- `alpha_p = 0.5`  
  Strength of polymer stress feedback:
  \[
  \tau_p = \alpha_p(C-I).
  \]
  Smaller values (e.g. `0.1`) give gentler coupling.

### Conformation equation
- `kappa_c = 1e-2`  
  Artificial diffusion on conformation components: \(+\kappa_c\Delta C\).
- `tauR = 1.0`  
  Relaxation time toward identity: \(-\frac{1}{\tau_R}(C-I)\).

### Time integration
- `dt_step = 5e-4`  
  Fixed time step.
- `t_end = 0.5`  
  Final simulation time.
- Conformation integrator: `d3.RK443`

### Diagnostics / printing
- Every 100 iterations:
  - `max(trC)` where `trC = cxx + cyy`
  - `max(|cxy|)`

### Output (snapshots)
A Dedalus file handler writes fields at regular **simulation time** intervals:
- `sim_dt = 0.1` (so writes at \(t=0.1,0.2,0.3,0.4,0.5\) given `t_end=0.5`)
- tasks written:
  - `ux`, `uy`, `p`, `cxx`, `cxy`, `cyy`

---

## Initial conditions

Conformation is initialized as identity plus a Gaussian “blob”:
\[
\text{blob}(x,y)=\exp\!\left(-\frac{(x-\pi)^2+(y-\pi/2)^2}{0.3^2}\right),
\]
\[
c_{xx}(x,y,0)=1+0.2\,\text{blob},\quad
c_{xy}(x,y,0)=0,\quad
c_{yy}(x,y,0)=1+0.2\,\text{blob}.
\]

---

## Notes / interpretation

- The Stokes part is **steady** (no inertia): we are effectively computing the instantaneous velocity field consistent with the current total forcing.
- The conformation dynamics are **nonlinear** due to advection \(u\cdot\nabla C\) and stretching \((\nabla u)C + C(\nabla u)^T\).
- Polymer feedback enters the flow only through the forcing:
  \[
  f_{\text{total}} = f_0 + \nabla\cdot(\alpha_p(C-I)).
  \]
 
---

## Quick sanity checks you can do

- Set `alpha_p = 0.0` → the Stokes velocity should revert to the fixed forcing solution and polymer no longer feeds back.
- Decrease `dt_step` and see whether output changes smoothly.
- Increase `kappa_c` if you see high-frequency noise / instability in `cxx,cxy,cyy`.

---
