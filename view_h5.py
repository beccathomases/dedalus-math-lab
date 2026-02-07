import h5py
import numpy as np
import matplotlib.pyplot as plt

fn = "snapshots-channel/snapshots-channel_s3.h5"

with h5py.File(fn, "r") as f:
    # list available tasks
    tasks = list(f["tasks"].keys())
    print("Tasks:", tasks)

    # pick one
    name = "tracer"   # tracer, pressure, u_x, u_z, vorticity
    A = f["tasks"][name][...]   # shape (nt, Nx, Nz)

    t = f["scales/sim_time"][...]
    it = f["scales/iteration"][...]

    # Find the coordinate datasets (they have hash names)
    x_key = [k for k in f["scales"].keys() if k.startswith("x_hash_")][0]
    z_key = [k for k in f["scales"].keys() if k.startswith("z_hash_")][0]
    x = f["scales"][x_key][...]   # length Nx
    z = f["scales"][z_key][...]   # length Nz

# choose a snapshot index (last one)
k = -1
Ak = A[k]   # shape (Nx, Nz)

plt.figure()
# pcolormesh expects 2D arrays shaped like (Nz, Nx) if we meshgrid; easiest is transpose
X, Z = np.meshgrid(x, z, indexing="xy")   # X,Z are (Nz, Nx)
plt.pcolormesh(X, Z, Ak.T, shading="auto")
plt.colorbar()
plt.title(f"{name}  iter={int(it[k])}  t={t[k]:.6g}")
plt.xlabel("x")
plt.ylabel("z")
plt.tight_layout()
plt.show()
