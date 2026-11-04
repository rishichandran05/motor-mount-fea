import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _surface_plot(nodes, tris, nodal_val, disp, scale, title, label, path, cmap="viridis"):
    pts = nodes + scale * disp
    tv = nodal_val[tris - 1].mean(axis=1)
    polys = pts[tris - 1]
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    pc = Poly3DCollection(polys, cmap=cmap, linewidths=0.05, edgecolors="k")
    pc.set_array(tv)
    ax.add_collection3d(pc)
    lo, hi = pts.min(0), pts.max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(22, -125)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
    fig.colorbar(pc, ax=ax, shrink=0.6, label=label)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return str(path)


def plot_mode(model, mode, path):
    U = model.modal["shapes"][mode]
    mag = np.linalg.norm(U, axis=1)
    scale = 0.15 * np.ptp(model.mesh["nodes"], axis=0).max() / max(mag.max(), 1e-12)
    f = model.modal["freqs"][mode - 1]
    return _surface_plot(model.mesh["nodes"], model.mesh["tris"], mag, U, scale,
                         f"Mode {mode}: {f:.0f} Hz (shape exaggerated)", "|u| (normalised)", path)


def plot_stress(model, path):
    s = model.static
    mag = np.linalg.norm(s["U"], axis=1)
    scale = 0.05 * np.ptp(model.mesh["nodes"], axis=0).max() / max(mag.max(), 1e-12)
    return _surface_plot(model.mesh["nodes"], model.mesh["tris"], s["vm_n"], s["U"], scale,
                         "von Mises stress (deformation exaggerated)", "MPa", path, cmap="inferno")
