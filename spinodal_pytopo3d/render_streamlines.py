"""
Stiff-axis orientation field as streamlines (like Senhora 2022 Fig. 5b/5e).

The optimized columnar high-stiffness axis per element is U[:,2] = stiff_axis(a,b,g).
We treat it as a (line) vector field on the element grid, sign-align it for coherent
integration, seed inside the solid region, and integrate streamlines that trace the
optimized orientation -- i.e. the principal-stress trajectories the columnar axis
aligns to. Streamlines are colored by local spinodal density (Frac), with a faint
translucent body for context, in a side (x1-x3) view matching the paper.

Usage:
  python -m spinodal_pytopo3d.render_streamlines spinodal_pytopo3d/results/cantilever_truss_64.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

from spinodal_pytopo3d import spinodal_material as sm
from spinodal_pytopo3d.export import _to_3d
from spinodal_pytopo3d.render_spinodal import (
    build_smooth_macro_field,
    _crop_png_whitespace,
    field_to_mesh,
    make_axis_coords,
    to_field_yxz,
)


def _vtk_order(flat, nx, ny, nz):
    """Flat Fortran element vector -> VTK ImageData point order (x fastest)."""
    return _to_3d(flat, nx, ny, nz).ravel(order="F")


def sign_align(axis):
    """Make the line field a coherent vector field: point toward +x1, with +x3
    as a tie-breaker for near-vertical members (so streamlines don't flip)."""
    ax = axis.astype(float).copy()
    ax[ax[:, 0] < 0] *= -1.0
    near_vert = np.abs(ax[:, 0]) < 0.3
    flip = near_vert & (ax[:, 2] < 0)
    ax[flip] *= -1.0
    return ax


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz")
    p.add_argument("--out", default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed-stride", type=int, default=0,
                   help="subsample seeds (0=auto)")
    p.add_argument("--n-seeds", type=int, default=250, help="approx number of seeds")
    p.add_argument("--smooth", type=float, default=1.2,
                   help="gaussian sigma (elements) to smooth the orientation field")
    p.add_argument("--tube", type=float, default=0.22, help="tube radius (elements)")
    p.add_argument("--color", default=None, help="single streamline color (else by density)")
    p.add_argument("--no-body", action="store_true", help="don't draw the faint body")
    p.add_argument("--presentation", action="store_true",
                   help="paper-style view: no axes, tighter zoom, cropped whitespace")
    p.add_argument("--interactive", action="store_true")
    args = p.parse_args()

    d = np.load(args.npz)
    nx, ny, nz = int(d["nelx"]), int(d["nely"]), int(d["nelz"])
    solid_flat = d["rho_bar"] >= args.threshold
    axis = sign_align(sm.stiff_axis(d["alpha"], d["beta"], d["gamma"]))  # (nele,3) flat F
    axis[~solid_flat] = 0.0                               # zero in void -> streamlines stop

    # smooth the (sign-aligned) field for long, coherent streamlines. Normalized
    # convolution (divide by smoothed solid weight) keeps void zeros from dragging
    # down solid values near member boundaries.
    if args.smooth > 0:
        from scipy import ndimage
        solid3 = _to_3d(solid_flat.astype(float), nx, ny, nz) > 0.5
        w = ndimage.gaussian_filter(solid3.astype(np.float32), args.smooth) + 1e-6
        comp = [ndimage.gaussian_filter(_to_3d(axis[:, k], nx, ny, nz), args.smooth) / w
                for k in range(3)]
        ax3 = np.stack(comp, axis=-1)
        nrm = np.linalg.norm(ax3, axis=-1, keepdims=True); nrm[nrm < 1e-9] = 1.0
        ax3 = ax3 / nrm
        ax3[~solid3] = 0.0
        vecs = np.column_stack([ax3[..., k].ravel(order="F") for k in range(3)])
    else:
        vecs = np.column_stack([_vtk_order(axis[:, k], nx, ny, nz) for k in range(3)])
    dens = _vtk_order(d["Frac"], nx, ny, nz)
    solid_vtk = _vtk_order(solid_flat.astype(float), nx, ny, nz) > 0.5

    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(1, 1, 1), origin=(0.5, 0.5, 0.5))
    grid["vectors"] = vecs
    grid["density"] = dens
    grid.set_active_vectors("vectors")

    # seeds: subsample solid element centers
    centers = grid.points[solid_vtk]
    n_solid = len(centers)
    stride = args.seed_stride or max(1, n_solid // max(1, args.n_seeds))
    seed = pv.PolyData(centers[::stride])
    print(f"{nx}x{ny}x{nz}: {n_solid} solid pts, {seed.n_points} seeds")

    streams = grid.streamlines_from_source(
        seed, vectors="vectors", integration_direction="both",
        max_steps=4000, max_length=4.0 * nx, initial_step_length=0.2,
        terminal_speed=1e-4,
    )
    print(f"streamlines: {streams.n_points:,} pts, {streams.n_lines:,} lines")
    tubes = streams.tube(radius=args.tube)

    # ---- render (side x1-x3 view, like Fig 5b/5e) ----
    pv.OFF_SCREEN = not args.interactive
    if args.interactive:
        win = [1400, 900]
    else:
        win = [2800, 950] if args.presentation else [2000, 1300]
    pl = pv.Plotter(off_screen=not args.interactive, window_size=win)
    pl.set_background("white")

    if not args.no_body:
        spc = 4
        xc = make_axis_coords(nx, spc, 1.0)
        yc = make_axis_coords(ny, spc, 1.0)
        zc = make_axis_coords(nz, spc, 1.0)
        mask_yxz = to_field_yxz(d["rho_bar"], nx, ny, nz) >= args.threshold
        try:
            sdf = build_smooth_macro_field(mask_yxz, xc, yc, zc, spc, 0.6, True)
            body = field_to_mesh(sdf, xc, yc, zc)
            pl.add_mesh(body, color="#cfe8e4", opacity=0.12, smooth_shading=True)
        except Exception as exc:
            print(f"[body] skipped ({exc})")

    if args.color:
        pl.add_mesh(tubes, color=args.color, smooth_shading=True)
    else:
        pl.add_mesh(tubes, scalars="density", cmap="plasma", clim=(0.3, 0.7),
                    smooth_shading=True,
                    scalar_bar_args=dict(title="Spinodal density", n_labels=3,
                                         vertical=True, position_x=0.88, height=0.5))
    pl.add_mesh(pv.Box(bounds=(0, nx, 0, ny, 0, nz)).extract_all_edges(),
                color="black", line_width=2)
    if not args.presentation:
        pl.add_axes(xlabel="x1", ylabel="x2", zlabel="x3", line_width=3)
    # side view: look along -x2, x1 horizontal, x3 up
    pl.camera_position = [(nx / 2, -3.0 * ny, nz / 2), (nx / 2, ny / 2, nz / 2), (0, 0, 1)]
    pl.camera.zoom(1.05 if args.presentation else 1.3)
    if not args.interactive:                      # SSAA framebuffer fails on-screen here
        try:
            pl.enable_anti_aliasing("ssaa")
        except Exception:
            pass

    if args.interactive:
        pl.show(title="Stiff-axis streamlines")
    else:
        out = args.out or args.npz.replace(".npz", "_streamlines.png")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        pl.screenshot(str(out))
        pl.close()
        if args.presentation:
            _crop_png_whitespace(out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
