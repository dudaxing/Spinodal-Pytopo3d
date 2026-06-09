"""
Export an optimized spinodal design (saved .npz) to STL and/or VTK.

* STL  — macro shape via PyTopo3D's marching-cubes `voxel_to_stl` (for printing/CAD).
* VTK  — legacy ASCII structured grid with per-cell scalars (V, Frac, rho_bar) and
         the columnar stiff-axis vector, so ParaView can render the density map and
         orientation *streamlines* (Glyph / Stream Tracer) — i.e. the paper's Fig. 5
         streamlines, but interactive.

Usage:
  python -m spinodal_pytopo3d.export spinodal_pytopo3d/results/cantilever_truss.npz
  python -m spinodal_pytopo3d.export <npz> --stl --vtk --thresh 0.5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "PyTopo3D-main"))

from spinodal_pytopo3d import spinodal_material as sm


def _to_3d(flat, nelx, nely, nelz):
    """Flat Fortran element vector (e=ely+elx*nely+elz*nelx*nely) -> arr[x,y,z]."""
    return np.asarray(flat).reshape(nelz, nelx, nely).transpose(1, 2, 0)


def _load(npz_path):
    d = np.load(npz_path)
    nelx, nely, nelz = int(d["nelx"]), int(d["nely"]), int(d["nelz"])
    axis = sm.stiff_axis(d["alpha"], d["beta"], d["gamma"])  # (nele,3)
    return d, nelx, nely, nelz, axis


def export_stl(npz_path, out_path=None, thresh=0.5, field="rho_bar",
               smooth_iterations=3):
    """Marching-cubes STL of the macro shape (presence field >= thresh)."""
    from pytopo3d.utils.export import voxel_to_stl

    d, nelx, nely, nelz, _ = _load(npz_path)
    vol = _to_3d(d[field], nelx, nely, nelz)
    out_path = out_path or npz_path.replace(".npz", ".stl")
    voxel_to_stl(vol.astype(float), out_path, level=thresh, padding=1,
                 smooth_mesh=smooth_iterations > 0, smooth_iterations=smooth_iterations)
    print(f"saved {out_path}")
    return out_path


def export_vtk(npz_path, out_path=None):
    """Legacy ASCII VTK (STRUCTURED_POINTS, cell data) with density + orientation."""
    d, nelx, nely, nelz, axis = _load(npz_path)
    nele = nelx * nely * nelz
    out_path = out_path or npz_path.replace(".npz", ".vtk")

    # VTK cell-data order is x-fastest; our fields are Fortran element order.
    def vtk_scalar(flat):
        return _to_3d(flat, nelx, nely, nelz).ravel(order="F")  # x fastest

    V = vtk_scalar(d["V"])
    Frac = vtk_scalar(d["Frac"])
    rho_bar = vtk_scalar(d["rho_bar"])
    ax = np.stack([_to_3d(axis[:, k], nelx, nely, nelz).ravel(order="F")
                   for k in range(3)], axis=1)  # (nele,3) in VTK order

    with open(out_path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("spinodal cantilever design\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nelx + 1} {nely + 1} {nelz + 1}\n")
        f.write("ORIGIN 0 0 0\n")
        f.write("SPACING 1 1 1\n")
        f.write(f"CELL_DATA {nele}\n")
        for name, arr in [("V", V), ("Frac", Frac), ("rho_bar", rho_bar)]:
            f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
            f.write("\n".join(f"{v:.5g}" for v in arr))
            f.write("\n")
        f.write("VECTORS stiff_axis float\n")
        f.write("\n".join(f"{x:.5g} {y:.5g} {z:.5g}" for x, y, z in ax))
        f.write("\n")
    print(f"saved {out_path}  (open in ParaView; Glyph/Stream Tracer on 'stiff_axis')")
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz")
    p.add_argument("--stl", action="store_true", help="export STL (default if neither given)")
    p.add_argument("--vtk", action="store_true", help="export VTK")
    p.add_argument("--thresh", type=float, default=0.5)
    p.add_argument("--field", default="rho_bar", choices=["rho_bar", "V"])
    args = p.parse_args()
    do_stl = args.stl or not (args.stl or args.vtk)
    if do_stl:
        export_stl(args.npz, thresh=args.thresh, field=args.field)
    if args.vtk:
        export_vtk(args.npz)


if __name__ == "__main__":
    main()
