import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob
import natsort

# -----------------------------
# 1. 获取所有 snapshot 文件
# -----------------------------
files = natsort.natsorted(glob.glob("snapshots/*.h5"))

# -----------------------------
# 2. 读取 snapshot 数据
# -----------------------------
tracer_list = []
pressure_list = []
vorticity_list = []

for file in files:
    with h5py.File(file, 'r') as f:
        tracer_list.append(f['tasks']['tracer'][0,:,:])
        pressure_list.append(f['tasks']['pressure'][0,:,:])
        vorticity_list.append(f['tasks']['vorticity'][0,:,:])

Nz, Nx = tracer_list[0].shape
n_snap = len(files)

# -----------------------------
# 3. 创建 figure
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(15,4))
titles = ['Tracer', 'Pressure', 'Vorticity']
cmaps = ['viridis', 'RdBu_r', 'plasma']

images = []
for ax, title, cmap, data_list in zip(axes, titles, cmaps, [tracer_list, pressure_list, vorticity_list]):
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    im = ax.imshow(data_list[0], origin='lower', aspect='auto', cmap=cmap)
    images.append(im)

plt.tight_layout()

# -----------------------------
# 4. 更新函数
# -----------------------------
t_total = 20  # 动画总时长 20 秒
def update(frame):
    for im, data_list in zip(images, [tracer_list, pressure_list, vorticity_list]):
        im.set_data(data_list[frame])
        im.set_clim(np.min(data_list[frame]), np.max(data_list[frame]))
    # 显示 t 从 0 -> 20 s
    t = frame / (n_snap - 1) * t_total
    fig.suptitle(f"t = {t:.2f} s")
    return images

# -----------------------------
# 5. 创建动画
# -----------------------------
# 计算每帧间隔，使总时长正好 20 秒
interval = 1000 * t_total / n_snap  # 毫秒
anim = FuncAnimation(fig, update, frames=n_snap, interval=interval, blit=False)

# -----------------------------
# 6. 保存动画
# -----------------------------
# fps = n_snap / 20 让播放总时间 20 s
fps = n_snap / t_total
anim.save('shear_flow_animation.mp4', fps=fps, dpi=150)
plt.show()
