import glob, os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ---- settings ----
snap_dir = "snapshots-channel"
field = "tracer"        # tracer, u_x, u_z, pressure, vorticity
out_dir = f"frames_{field}"
os.makedirs(out_dir, exist_ok=True)

# collect snapshot files in order
files = sorted(glob.glob(os.path.join(snap_dir, "*.h5")))
if not files:
    raise RuntimeError(f"No .h5 files found in {snap_dir}")

frame = 0

for fn in files:
    with h5py.File(fn, "r") as f:
        A = f["tasks"][field][...]          # shape (nt, Nx, Nz)
        t = f["scales/sim_time"][...]

        # coordinates (hashed names)
        x_key = [k for k in f["scales"].keys() if k.startswith("x_hash_")][0]
        z_key = [k for k in f["scales"].keys() if k.startswith("z_hash_")][0]
        x = f["scales"][x_key][...]
        z = f["scales"][z_key][...]

    # Make mesh once per file
    X, Z = np.meshgrid(x, z, indexing="xy")   # (Nz, Nx)

    for k in range(A.shape[0]):
        Ak = A[k]                             # (Nx, Nz)

        plt.figure(figsize=(6, 3))
        plt.pcolormesh(X, Z, Ak.T, shading="auto")
        plt.colorbar()
        plt.title(f"{field}   t={t[k]:.4f}")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.tight_layout()

        png = os.path.join(out_dir, f"frame_{frame:05d}.png")
        plt.savefig(png, dpi=150)
        plt.close()
        frame += 1

print(f"Wrote {frame} frames to {out_dir}/")
