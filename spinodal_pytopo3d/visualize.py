"""
3D visualization of an optimized spinodal cantilever (saved .npz).

Produces, for elements with physical volume above a threshold:
  * a voxel plot colored by spinodal density (Frac), and
  * a quiver of the columnar stiff axis (local e3 in global frame),
which together mirror Fig. 5 of Senhora 2022 (material + streamlines).

Usage:  python -m spinodal_pytopo3d.visualize spinodal_pytopo3d/results/cantilever_truss.npz
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

from spinodal_pytopo3d import spinodal_material as sm


def _grid_index(nelx, nely, nelz):
    """Map flat (Fortran e = ely + elx*nely + elz*nelx*nely) -> (x,y,z) centers."""
    e = np.arange(nelx * nely * nelz)
    ely = e % nely
    elx = (e // nely) % nelx
    elz = e // (nelx * nely)
    return elx, ely, elz


def visualize(npz_path, thresh=0.5, quiver_stride=1, out_path=None):
    d = np.load(npz_path)
    nelx, nely, nelz = int(d["nelx"]), int(d["nely"]), int(d["nelz"])
    V, Frac = d["V"], d["Frac"]
    alpha, beta, gamma = d["alpha"], d["beta"], d["gamma"]
    elx, ely, elz = _grid_index(nelx, nely, nelz)

    show = V >= thresh * V.max() if V.max() > 0 else V > 1e9
    fig = plt.figure(figsize=(13, 5))

    # (a) density voxels
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    vox = np.zeros((nelx, nely, nelz), dtype=bool)
    col = np.zeros(vox.shape + (4,))
    cmap = cm.get_cmap("plasma")
    fr_norm = (Frac - 0.3) / 0.4
    for i in np.where(show)[0]:
        vox[elx[i], ely[i], elz[i]] = True
        col[elx[i], ely[i], elz[i]] = cmap(np.clip(fr_norm[i], 0, 1))
    ax.voxels(vox, facecolors=col, edgecolor=(0, 0, 0, 0.06))
    ax.set_title(f"spinodal density (Frac)  |  shown V>={thresh:.2f}·max")
    ax.set_box_aspect((nelx, nely, nelz))
    m = cm.ScalarMappable(cmap=cmap); m.set_array([0.3, 0.7])
    fig.colorbar(m, ax=ax, shrink=0.5, label="Frac")

    # (b) stiff-axis quiver
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    axis = sm.stiff_axis(alpha, beta, gamma)  # (nele,3)
    sel = np.where(show)[0][::quiver_stride]
    cx, cy, cz = elx[sel] + 0.5, ely[sel] + 0.5, elz[sel] + 0.5
    u, v, w = axis[sel, 0], axis[sel, 1], axis[sel, 2]
    ax2.quiver(cx, cy, cz, u, v, w, length=1.2, normalize=True,
               color="tab:blue", linewidth=0.6)
    ax2.set_xlim(0, nelx); ax2.set_ylim(0, nely); ax2.set_zlim(0, nelz)
    ax2.set_box_aspect((nelx, nely, nelz))
    ax2.set_title("columnar stiff-axis orientation")

    ratio = float(d["ratio"]) if "ratio" in d else float("nan")
    fig.suptitle(f"{os.path.basename(npz_path)}   f/f0 = {ratio:.3f}", fontsize=12)
    fig.tight_layout()
    out_path = out_path or npz_path.replace(".npz", "_design.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved {out_path}")
    return out_path


if __name__ == "__main__":
    visualize(sys.argv[1] if len(sys.argv) > 1 else
              "spinodal_pytopo3d/results/cantilever_truss.npz")
