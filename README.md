# Dedalus Math Lab

This repository contains **Dedalus simulations** for studying **periodic shear flows** and other fluid dynamics problems. It is intended for research and educational purposes in computational fluid dynamics (CFD).

## Features

- **Periodic shear flow simulations** with configurable domain and shear profiles
- **Fourier–Chebyshev discretization**: periodic in x, non-periodic in z
- Time evolution using Dedalus IVP solver with optional viscosity and damping
- Scripts for initializing velocity fields, solving diffusion equations, and visualizing results
- Example plotting of **Tracer, Pressure, and Vorticity fields** with animation support

## Requirements

- Python 3.8+  
- [Dedalus](http://dedalus-project.org/)  
- NumPy, Matplotlib  
- h5py (for snapshot data)  

Install dependencies (recommended in a virtual environment):

```bash
pip install dedalus numpy matplotlib h5py
```

Activate environments before running scripts:
```bash
conda activate dedalus
```

## Directory Structure

```
dedalus-math-lab/
├─ main # main branch, containing README.md and Tutorials
├─ periodic shear flow # Branch 1
├─ poisson equation # Branch 2
```
