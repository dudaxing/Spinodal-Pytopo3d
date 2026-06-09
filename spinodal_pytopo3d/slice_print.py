"""
Slice-based binary images for resin (DLP/LCD/SLA) printing of the embedded
columnar spinodal cantilever -- the paper's approach (no STL).

The full 3D microstructure at print resolution is billions of voxels, so we never
hold it in memory: each print layer is evaluated as a 2D binary image at the
printer's XY pixel resolution.

  cured(x, y) = inside_macro_envelope(x, y, z)  AND  phi(x, y, z) >= cutoff(rho_local)

Because the pixel pitch is far below the pore size, the spinodal is well resolved
and the binary slice is clean (unlike the under-sampled marching-cubes render).

Sample mode (`--sample N`) renders N representative layers at full print resolution
plus a montage, and reports the full-stack sizing -- to preview before committing.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

from spinodal_pytopo3d.render_spinodal import (
    build_smooth_macro_field,
    build_spinodal_field,
    global_phases,
    keep_largest_component,
    load_npz_fields,
    m_from_gamma,
    make_axis_coords,
    nearest_material_index_map,
    owner_indices,
    owner_ranges,
    rotation,
    sample_columnar_wave_vectors_z,
)


def _macro_sdf_coarse(mask, spc, padding, sigma_cells):
    ny, nx, nz = mask.shape
    xc = make_axis_coords(nx, spc, padding)
    yc = make_axis_coords(ny, spc, padding)
    zc = make_axis_coords(nz, spc, padding)
    sdf = build_smooth_macro_field(mask, xc, yc, zc, spc, sigma_cells, True)
    return sdf, (xc, yc, zc)  # sdf shape (len xc, len yc, len zc) = (x,y,z)


def _interp_inside(sdf, coords, xpx, ypx, zc_elem):
    xc, yc, zc = coords
    sx, sy, sz = xc[1] - xc[0], yc[1] - yc[0], zc[1] - zc[0]
    ix = (xpx - xc[0]) / sx
    iy = (ypx - yc[0]) / sy
    iz = (zc_elem - zc[0]) / sz
    II, JJ = np.meshgrid(ix, iy, indexing="ij")
    KK = np.full(II.shape, iz)
    val = ndi.map_coordinates(sdf, [II, JJ, KK], order=1, mode="nearest")
    return val < 0.0


def _interp_keep(keep, coords, xpx, ypx, zc_elem):
    xc, yc, zc = coords
    sx, sy, sz = xc[1] - xc[0], yc[1] - yc[0], zc[1] - zc[0]
    II, JJ = np.meshgrid((xpx - xc[0]) / sx, (ypx - yc[0]) / sy, indexing="ij")
    KK = np.full(II.shape, (zc_elem - zc[0]) / sz)
    return ndi.map_coordinates(keep.astype(np.float32), [II, JJ, KK],
                               order=0, mode="nearest") > 0.5


def slice_layer(zc_elem, xpx, ypx, fields, nearest, base, kappa, n_waves, phases,
                sdf, sdf_coords, keep=None, keep_coords=None):
    ny, nx, nz = fields["Frac"].shape
    iz = int(np.clip(np.floor(zc_elem), 0, nz - 1))
    ixo, iyo = owner_indices(xpx, nx), owner_indices(ypx, ny)
    xr, yr = owner_ranges(ixo, nx), owner_ranges(iyo, ny)
    micro = np.full((len(xpx), len(ypx)), -1.0, np.float32)
    frac, al, be, ga = fields["Frac"], fields["alpha"], fields["beta"], fields["gamma"]
    import scipy.special as special
    zg = np.float32(zc_elem)                          # GLOBAL z (seamless across layers)
    for ix in range(nx):
        xs = xr[ix]
        if xs.start == xs.stop:
            continue
        xg = xpx[xs]                                  # GLOBAL coords
        for iy in range(ny):
            ys = yr[iy]
            if ys.start == ys.stop:
                continue
            yg = ypx[ys]
            py, px, pz = nearest[:, iy, ix, iz]
            fr = float(np.clip(frac[py, px, pz], 1e-5, 1 - 1e-5))
            cut = np.float32(np.sqrt(2.0) * special.erfinv(2 * fr - 1))
            R = rotation(float(al[py, px, pz]), float(be[py, px, pz]), float(ga[py, px, pz]))
            vecs = base @ R.T
            XX, YY = np.meshgrid(xg, yg, indexing="ij")
            pts = np.column_stack((XX.ravel(), YY.ravel(),
                                   np.full(XX.size, zg, np.float32))).astype(np.float32)
            ang = pts @ vecs.T
            ang *= kappa; ang += phases[None, :]; np.cos(ang, out=ang)
            phi = ang.sum(1).astype(np.float32) * np.float32(np.sqrt(2.0 / n_waves))
            micro[xs, ys] = (phi - cut).reshape((len(xg), len(yg)))
    inside = _interp_inside(sdf, sdf_coords, xpx, ypx, zc_elem)
    cured = (micro >= 0) & inside
    if keep is not None:
        cured &= _interp_keep(keep, keep_coords, xpx, ypx, zc_elem)
    return cured


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz")
    p.add_argument("--outdir", default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--element-mm", type=float, default=3.0, help="cubic element size (mm)")
    p.add_argument("--pixel-um", type=float, default=50.0, help="printer XY pixel pitch")
    p.add_argument("--layer-um", type=float, default=50.0, help="printer layer height")
    p.add_argument("--m", type=float, default=4.0, help="pore frequency (pore~element/m)")
    p.add_argument("--gamma", type=float, default=None,
                   help="pore wavenumber gamma (cm^-1); overrides --m (paper cantilever: 6)")
    p.add_argument("--n-waves", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260608)
    p.add_argument("--macro-spc", type=int, default=6)
    p.add_argument("--macro-sigma", type=float, default=0.65)
    p.add_argument("--padding", type=float, default=1.0)
    p.add_argument("--sample", type=int, default=8, help="N representative layers (0=full stack)")
    p.add_argument("--declutter", action="store_true",
                   help="remove floating bits via 3D connected components (keep largest)")
    p.add_argument("--declutter-spc", type=int, default=6,
                   help="cleanup-resolution samples/cell for the 3D connectivity pass")
    args = p.parse_args()

    fields, nx, ny, nz, raw = load_npz_fields(args.npz)
    mask = fields["rho_bar"] >= args.threshold
    nearest = nearest_material_index_map(mask)

    m_eff = m_from_gamma(args.gamma, args.element_mm) if args.gamma is not None else args.m
    h, px, lh = args.element_mm, args.pixel_um / 1000.0, args.layer_um / 1000.0
    Lx, Ly, Lz = nx * h, ny * h, nz * h                 # part size (mm)
    nxpx, nypx, nlayers = round(Lx / px), round(Ly / px), round(Lz / lh)
    pore_mm = h / m_eff

    if args.gamma is not None:
        print(f"gamma={args.gamma}/cm -> m={m_eff:.3f}")
    print(f"part: {Lx:.1f} x {Ly:.1f} x {Lz:.1f} mm  (cubic element {h} mm)")
    print(f"printer: pixel {args.pixel_um:g} um, layer {args.layer_um:g} um")
    print(f"full stack: {nlayers} layers of {nxpx} x {nypx} px")
    print(f"pore size ~ {pore_mm*1000:.0f} um  ({pore_mm/px:.0f} px/pore)")
    onebit = nxpx * nypx / 8
    print(f"est. full stack (1-bit, uncompressed): {onebit*nlayers/1e6:.0f} MB "
          f"(PNG-compressed typically far smaller)")
    if pore_mm / px < 3:
        print("WARNING: pore < 3 px -> may not print cleanly; lower --m or finer pixel.")

    # element-unit print coordinates
    xpx = (np.arange(nxpx) + 0.5) / nxpx * nx
    ypx = (np.arange(nypx) + 0.5) / nypx * ny
    base = sample_columnar_wave_vectors_z(args.n_waves, args.seed)
    phases = global_phases(args.seed, args.n_waves)
    kappa = np.float32(2 * np.pi * m_eff)
    sdf, sdf_coords = _macro_sdf_coarse(mask, args.macro_spc, args.padding, args.macro_sigma)

    keep, keep_coords = None, None
    if args.declutter:
        spc_c = args.declutter_spc
        xcc = make_axis_coords(nx, spc_c, args.padding)
        ycc = make_axis_coords(ny, spc_c, args.padding)
        zcc = make_axis_coords(nz, spc_c, args.padding)
        macro_c = build_smooth_macro_field(mask, xcc, ycc, zcc, spc_c, args.macro_sigma, True)
        micro_c = build_spinodal_field(mask, fields["Frac"], fields["alpha"], fields["beta"],
                                       fields["gamma"], xcc, ycc, zcc, m_eff, args.n_waves,
                                       args.seed, phases)
        keep, ncomp, nkept = keep_largest_component(np.maximum(micro_c, macro_c) >= 0, 26)
        keep_coords = (xcc, ycc, zcc)
        print(f"declutter (cleanup spc={spc_c}): {ncomp} components -> kept largest "
              f"({int(keep.sum()):,}/{int((np.maximum(micro_c,macro_c)>=0).sum()):,} voxels). "
              "NOTE: macro-scale floaters removed; sub-cleanup-resolution specks may remain.")

    outdir = Path(args.outdir or Path(args.npz).with_suffix(""))
    outdir = Path(str(outdir) + "_slices")
    outdir.mkdir(parents=True, exist_ok=True)

    if args.sample and args.sample > 0:
        layers = np.linspace(0.15, 0.85, args.sample) * nlayers
        layers = sorted(set(int(round(v)) for v in layers))
    else:
        layers = range(nlayers)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imgs, t0 = [], time.time()
    for li in layers:
        zc_elem = (li + 0.5) / nlayers * nz
        cured = slice_layer(zc_elem, xpx, ypx, fields, nearest, base, kappa,
                            args.n_waves, phases, sdf, sdf_coords, keep, keep_coords)
        frac_cure = float(cured.mean())
        fn = outdir / f"layer_{li:04d}.png"
        plt.imsave(fn, cured.T[::-1], cmap="gray", vmin=0, vmax=1)  # T->(row=y); flip y up
        imgs.append((li, cured, frac_cure))
        print(f"  layer {li:4d}/{nlayers} z={zc_elem:.2f}el  cured={frac_cure*100:.1f}%  "
              f"{time.time()-t0:.1f}s -> {fn.name}")

    # montage
    n = len(imgs)
    cols = min(4, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2 * Ly / Lx))
    axes = np.atleast_1d(axes).ravel()
    for ax, (li, cured, fc) in zip(axes, imgs):
        ax.imshow(cured.T[::-1], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"layer {li}  ({fc*100:.0f}% cured)", fontsize=8)
        ax.axis("off")
    for ax in axes[len(imgs):]:
        ax.axis("off")
    fig.suptitle(f"Spinodal print slices (sample) — pore~{pore_mm*1000:.0f}um, "
                 f"{nxpx}x{nypx}px/layer, {nlayers} layers", fontsize=10)
    fig.tight_layout()
    montage = outdir / "_sample_montage.png"
    fig.savefig(montage, dpi=130)
    plt.close(fig)
    print(f"saved sample slices + montage in {outdir}")


if __name__ == "__main__":
    main()
