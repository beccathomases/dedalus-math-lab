import h5py
import numpy as np
import matplotlib.pyplot as plt

fn = "snapshots-channel/snapshots-channel_s1.h5"  # change if needed

with h5py.File(fn, "r") as f:
    # Dedalus commonly stores tasks under /tasks
    print("Tasks:", list(f["tasks"].keys()))
    t = f["scales/sim_time"][:] if "scales" in f and "sim_time" in f["scales"] else None

    # choose what you want to plot:
    name = "vorticity"  # or 'tracer', 'u_x', 'u_z', 'pressure'
    data = f["tasks"][name][:]     # typically shape (nt, Nx, Nz) or (nt, Nz, Nx)

    # grab the last saved time
    A = data[-1]

# Make a best guess about orientation; if it looks rotated, transpose A below
plt.figure()
plt.imshow(A.T, origin="lower", aspect="auto")  # try A or A.T depending on file layout
plt.colorbar()
plt.title(f"{name} (last snapshot)")
plt.xlabel("x index")
plt.ylabel("z index")
plt.tight_layout()
plt.show()
