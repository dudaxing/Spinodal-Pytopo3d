"""
Embed and render the columnar spinodal microstructure for an optimized cantilever
(reproducing Senhora 2022 Fig. 5a/d-style renderings), adapted from the repo's
`render_cloaking_smooth_spinodal.py` to consume this package's .npz results.

Pipeline:
  1. macro envelope  : rho_bar >= thresh -> signed-distance field -> smoothed level set
  2. micro embedding : per coarse cell, columnar GRF with LOCAL density (Frac) and
                       LOCAL orientation R(alpha,beta,gamma); solid where phi >= cutoff
  3. union -> marching cubes -> PyVista render (teal), PNG (+ optional PLY)

Key vs the cloak script:
  * reads .npz, takes nelx/nely/nelz from the file (element order already matches),
  * macro mask uses rho_bar (macro presence), not V=rho_bar*Frac (which is <=0.7),
  * COLUMNAR AXIS = local z (this package's stiff axis is U[:,2], D33 stiffest),
    so wave vectors are sampled near the x-y plane (columns run along z).

Usage (from project root, after `pip install pyvista`):
  python -m spinodal_pytopo3d.render_spinodal spinodal_pytopo3d/results/cantilever_truss.npz \
      --samples-per-cell 8 --m 5 --n-waves 64
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import scipy.special as special
from skimage import measure


# ---------------------------------------------------------------- field loading
def to_field_yxz(values, nx, ny, nz):
    """Flat element order (e=iy+ix*ny+iz*nx*ny) -> (y,x,z) array."""
    field = np.empty((ny, nx, nz), dtype=np.float32)
    v = np.asarray(values, dtype=np.float32).ravel()
    for iz in range(nz):
        for ix in range(nx):
            start = iz * nx * ny + ix * ny
            field[:, ix, iz] = v[start:start + ny]
    return field


def load_npz_fields(npz_path):
    d = np.load(npz_path)
    nx, ny, nz = int(d["nelx"]), int(d["nely"]), int(d["nelz"])
    fields = {k: to_field_yxz(d[k], nx, ny, nz)
              for k in ("rho_bar", "Frac", "alpha", "beta", "gamma")}
    return fields, nx, ny, nz, d


# ---------------------------------------------------------------- rotations
def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def rotation(alpha, beta, gamma):
    return _rx(gamma) @ _ry(beta) @ _rz(alpha)   # U = Rx(g)Ry(b)Rz(a), matches FEA


def sample_columnar_wave_vectors_z(count, seed, cone_deg=30.0, leakage=0.10):
    """Wave vectors near the x-y plane -> structure invariant along z (columns||z)."""
    rng = np.random.default_rng(seed)
    plane_sin = math.sin(math.radians(cone_deg))
    out = []
    while len(out) < count:
        v = rng.normal(size=3).astype(np.float32)
        n = np.linalg.norm(v)
        if n == 0:
            continue
        v /= n
        if (rng.random() < leakage) or (abs(v[2]) < plane_sin):  # near x-y plane
            out.append(v)
    return np.stack(out, axis=0)


def phases_for_cell(seed, iy, ix, iz, count):
    cs = (seed + 73856093 * (ix + 1) + 19349663 * (iy + 1) + 83492791 * (iz + 1)) & 0xFFFFFFFF
    return np.random.default_rng(cs).uniform(0, 2 * np.pi, count).astype(np.float32)


def global_phases(seed, n_waves):
    """One shared phase set for the whole domain (paper Eq S31). Combined with
    GLOBAL coordinates this yields a seamless field -- per-cell random phases
    (phases_for_cell) instead produce a seam at every cell boundary."""
    return np.random.default_rng(seed + 1).uniform(0, 2 * np.pi, n_waves).astype(np.float32)


def m_from_gamma(gamma_per_cm, element_mm):
    """Pore frequency m (element units) from physical wavenumber gamma (cm^-1):
    physical wavelength = 2*pi/gamma; m = element_size / wavelength."""
    return gamma_per_cm * (element_mm / 10.0) / (2.0 * np.pi)


# ---------------------------------------------------------------- grids
def make_axis_coords(length_cells, spc, padding):
    spacing = 1.0 / spc
    start, stop = -padding, length_cells + padding
    count = int(round((stop - start) / spacing)) + 1
    return (start + spacing * np.arange(count, dtype=np.float32)).astype(np.float32)


def owner_indices(coords, n):
    return np.clip(np.floor(coords).astype(np.int32), 0, n - 1)


def owner_ranges(owners, n):
    out = []
    for i in range(n):
        hit = np.flatnonzero(owners == i)
        out.append(slice(int(hit[0]), int(hit[-1]) + 1) if hit.size else slice(0, 0))
    return out


# ---------------------------------------------------------------- macro envelope
def build_smooth_macro_field(mask, xc, yc, zc, spc, sigma_cells, volume_correct):
    ny, nx, nz = mask.shape
    ixo, iyo, izo = owner_indices(xc, nx), owner_indices(yc, ny), owner_indices(zc, nz)
    inside = ((xc[:, None, None] >= 0) & (xc[:, None, None] <= nx)
              & (yc[None, :, None] >= 0) & (yc[None, :, None] <= ny)
              & (zc[None, None, :] >= 0) & (zc[None, None, :] <= nz))
    occ = mask[iyo[None, :, None], ixo[:, None, None], izo[None, None, :]] & inside
    spacing = 1.0 / spc
    sdf = (ndi.distance_transform_edt(~occ, sampling=(spacing,) * 3)
           - ndi.distance_transform_edt(occ, sampling=(spacing,) * 3)).astype(np.float32)
    sig = max(0.0, sigma_cells * spc)
    if sig > 0:
        sdf = ndi.gaussian_filter(sdf, sigma=sig, mode="nearest").astype(np.float32)
    if volume_correct:
        tgt = int(occ.sum())
        if 0 < tgt < sdf.size:
            sdf -= np.float32(np.partition(sdf.ravel(), tgt - 1)[tgt - 1])
    return sdf


def nearest_material_index_map(mask):
    if not mask.any():
        raise ValueError("No material cells with current threshold.")
    _, idx = ndi.distance_transform_edt(~mask, return_indices=True)
    return idx.astype(np.int16)


# ---------------------------------------------------------------- micro embedding
def build_spinodal_field(mask, frac, alpha, beta, gamma, xc, yc, zc, m, n_waves, seed,
                         phases=None):
    ny, nx, nz = mask.shape
    ixo, iyo, izo = owner_indices(xc, nx), owner_indices(yc, ny), owner_indices(zc, nz)
    xr, yr, zr = owner_ranges(ixo, nx), owner_ranges(iyo, ny), owner_ranges(izo, nz)
    nearest = nearest_material_index_map(mask)
    base = sample_columnar_wave_vectors_z(n_waves, seed)
    if phases is None:
        phases = global_phases(seed, n_waves)        # one global set -> seamless
    kappa = np.float32(2 * np.pi * m)
    micro = np.empty((len(xc), len(yc), len(zc)), dtype=np.float32)
    total, done, t0 = nx * ny * nz, 0, time.time()
    step = max(1, total // 12)
    for ix in range(nx):
        xs = xr[ix]; xg = xc[xs]                      # GLOBAL coords (no per-cell offset)
        for iy in range(ny):
            ys = yr[iy]; yg = yc[ys]
            for iz in range(nz):
                zs = zr[iz]; zg = zc[zs]
                py, px, pz = nearest[:, iy, ix, iz]
                fr = float(np.clip(frac[py, px, pz], 1e-5, 1 - 1e-5))
                cut = np.float32(np.sqrt(2.0) * special.erfinv(2 * fr - 1))
                R = rotation(float(alpha[py, px, pz]), float(beta[py, px, pz]),
                             float(gamma[py, px, pz]))
                vecs = base @ R.T
                xx, yy, zz = np.meshgrid(xg, yg, zg, indexing="ij")
                pts = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())).astype(np.float32)
                ang = pts @ vecs.T
                ang *= kappa; ang += phases[None, :]; np.cos(ang, out=ang)
                phi = ang.sum(axis=1).astype(np.float32) * np.float32(np.sqrt(2.0 / n_waves))
                micro[xs, ys, zs] = (phi - cut).reshape((len(xg), len(yg), len(zg)))
                done += 1
                if done == 1 or done % step == 0 or done == total:
                    print(f"  micro [{done:4d}/{total}] {time.time()-t0:.1f}s")
    return micro


# ---------------------------------------------------------------- de-floating
def keep_largest_component(solid, connectivity=26, min_frac=0.0):
    """Voxel connected-component cleanup (remove floating bits).

    Keeps the largest solid body; if min_frac>0, also keeps any component whose
    size >= min_frac * largest. Returns (kept_mask, n_components, n_kept).
    Uses scipy.ndimage.label (linear-time union-find CCL).
    """
    from scipy import ndimage
    struct = (np.ones((3, 3, 3), dtype=bool) if connectivity == 26
              else ndimage.generate_binary_structure(3, 1))  # 6-conn otherwise
    labels, n = ndimage.label(solid, structure=struct)
    if n <= 1:
        return solid, n, n
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    if min_frac > 0.0:
        keep_ids = np.where(sizes >= min_frac * sizes.max())[0]
        return np.isin(labels, keep_ids), n, int(keep_ids.size)
    return labels == int(sizes.argmax()), n, 1


# ---------------------------------------------------------------- mesh + render
def field_to_mesh(field, xc, yc, zc):
    import pyvista as pv
    sp = (float(xc[1]-xc[0]), float(yc[1]-yc[0]), float(zc[1]-zc[0]))
    if not (field.min() <= 0 <= field.max()):
        raise RuntimeError(f"field does not cross zero ({field.min()},{field.max()})")
    verts, faces, _, _ = measure.marching_cubes(field, level=0.0, spacing=sp)
    verts += np.array([xc[0], yc[0], zc[0]], dtype=np.float32)
    pvf = np.empty((faces.shape[0], 4), dtype=np.int64)
    pvf[:, 0] = 3; pvf[:, 1:] = faces
    return pv.PolyData(verts.astype(np.float32), pvf.ravel())


def density_scalars(mesh, frac_yxz, nx, ny, nz):
    """Per-vertex local spinodal density (Frac) by nearest coarse cell."""
    pts = np.asarray(mesh.points)
    ix = np.clip(np.floor(pts[:, 0]).astype(int), 0, nx - 1)
    iy = np.clip(np.floor(pts[:, 1]).astype(int), 0, ny - 1)
    iz = np.clip(np.floor(pts[:, 2]).astype(int), 0, nz - 1)
    return frac_yxz[iy, ix, iz]


def render(mesh, out_png, nx, ny, nz, color="#4DB6AC", box=True, size=(2200, 1500),
           annotate=False, scalars=None, title=None):
    import pyvista as pv
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=list(size))
    pl.set_background("white")
    if scalars is not None:
        pl.add_mesh(mesh, scalars=scalars, cmap="viridis", clim=(0.3, 0.7),
                    smooth_shading=True, specular=0.5, specular_power=30, ambient=0.3,
                    scalar_bar_args=dict(title="Spinodal density", n_labels=3,
                                         vertical=True, position_x=0.86, height=0.5))
    else:
        pl.add_mesh(mesh, color=color, smooth_shading=True, specular=0.5,
                    specular_power=30, ambient=0.3)
    if box:
        pl.add_mesh(pv.Box(bounds=(0, nx, 0, ny, 0, nz)).extract_all_edges(),
                    color="black", line_width=2)
    if annotate:
        # load arrow at the free end (x=nx), pointing -z (magenta, like the paper)
        alen = 0.45 * nz
        pl.add_mesh(pv.Arrow(start=(nx, ny / 2, nz + alen), direction=(0, 0, -1),
                             scale=alen, tip_length=0.3, shaft_radius=0.04,
                             tip_radius=0.09), color="magenta")
        pl.add_point_labels([(nx, ny / 2, nz + alen)], ["Load"], font_size=26,
                            text_color="magenta", shape=None, show_points=False)
        pl.add_axes(xlabel="x1", ylabel="x2", zlabel="x3", line_width=3)
        if title:
            pl.add_text(title, position="upper_left", font_size=18, color="black")
    try:
        pl.enable_eye_dome_lighting()
        pl.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    pl.camera_position = [(1.9 * nx, -1.5 * ny, 1.25 * nz),
                          (nx / 2, ny / 2, nz / 2), (0, 0, 1)]
    pl.reset_camera()
    pl.camera.zoom(0.9)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(str(out_png))
    pl.close()
    print(f"saved {out_png}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz")
    p.add_argument("--out", default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--samples-per-cell", type=int, default=8)
    p.add_argument("--m", type=float, default=5.0, help="pore frequency in element units")
    p.add_argument("--gamma", type=float, default=None,
                   help="pore wavenumber gamma (cm^-1); overrides --m (paper cantilever: 6)")
    p.add_argument("--element-mm", type=float, default=3.0,
                   help="cubic element size (mm), used with --gamma")
    p.add_argument("--n-waves", type=int, default=64)
    p.add_argument("--macro-sigma", type=float, default=0.65)
    p.add_argument("--padding", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260608)
    p.add_argument("--envelope-only", action="store_true")
    p.add_argument("--save-ply", action="store_true")
    p.add_argument("--save-stl", action="store_true", help="binary STL for mesh viewers")
    p.add_argument("--no-png", action="store_true", help="skip the rendered PNG")
    p.add_argument("--annotate", action="store_true", help="load arrow, axes, f/f0 label")
    p.add_argument("--color-by-density", action="store_true", help="color surface by Frac")
    p.add_argument("--declutter", action="store_true",
                   help="remove floating bits via 3D connected components (keep largest)")
    p.add_argument("--declutter-min-frac", type=float, default=0.0,
                   help="also keep components >= this fraction of the largest (0=largest only)")
    args = p.parse_args()

    fields, nx, ny, nz, raw = load_npz_fields(args.npz)
    mask = fields["rho_bar"] >= args.threshold
    print(f"mesh {nx}x{ny}x{nz}; material cells (rho_bar>={args.threshold}): "
          f"{int(mask.sum())}/{mask.size}; Frac[mask] "
          f"min={float(fields['Frac'][mask].min()):.3f} max={float(fields['Frac'][mask].max()):.3f}")

    spc = max(4, args.samples_per_cell)
    xc = make_axis_coords(nx, spc, args.padding)
    yc = make_axis_coords(ny, spc, args.padding)
    zc = make_axis_coords(nz, spc, args.padding)
    print(f"fine grid (x,y,z) = {len(xc)}x{len(yc)}x{len(zc)} = {len(xc)*len(yc)*len(zc):,} voxels")

    m_eff = args.m
    if args.gamma is not None:
        m_eff = m_from_gamma(args.gamma, args.element_mm)
        print(f"gamma={args.gamma}/cm, element={args.element_mm}mm -> m={m_eff:.3f} "
              f"(physical wavelength {2*np.pi/args.gamma*10:.2f} mm)")
    phases = global_phases(args.seed, args.n_waves)

    macro = build_smooth_macro_field(mask, xc, yc, zc, spc, args.macro_sigma, True)
    if args.envelope_only:
        final = macro
    else:
        micro = build_spinodal_field(mask, fields["Frac"], fields["alpha"],
                                     fields["beta"], fields["gamma"], xc, yc, zc,
                                     m_eff, args.n_waves, args.seed, phases)
        final = np.maximum(micro, macro)

    if args.declutter and not args.envelope_only:
        solid = final >= 0
        kept, ncomp, nkept = keep_largest_component(solid, 26, args.declutter_min_frac)
        removed = int((solid & ~kept).sum())
        final = np.where(kept, final, np.float32(-1.0)).astype(np.float32)
        print(f"declutter: {ncomp} components -> kept {nkept} "
              f"(removed {removed:,} solid voxels, {removed/max(int(solid.sum()),1)*100:.1f}%)")

    mesh = field_to_mesh(final, xc, yc, zc)
    print(f"mesh: {mesh.n_points:,} pts, {mesh.n_cells:,} faces")

    out = args.out or args.npz.replace(".npz", "_spinodal.png")
    if args.save_ply:
        mesh.save(str(Path(out).with_suffix(".ply")))
        print(f"saved {Path(out).with_suffix('.ply')}")
    if args.save_stl:
        stl_path = Path(out).with_suffix(".stl")
        mesh.save(str(stl_path), binary=True)
        mb = stl_path.stat().st_size / 1e6
        print(f"saved {stl_path}  ({mb:.0f} MB binary STL, {mesh.n_cells:,} triangles)")
        print("  view in MeshLab / Blender / Windows 3D Viewer / ParaView "
              "(CAD software may struggle with this triangle count).")
    if args.no_png:
        return
    scal = density_scalars(mesh, fields["Frac"], nx, ny, nz) if args.color_by_density else None
    title = None
    if args.annotate and "ratio" in raw.files and np.isfinite(float(raw["ratio"])):
        title = f"f/f0 = {float(raw['ratio']):.2f}"
    render(mesh, out, nx, ny, nz, annotate=args.annotate, scalars=scal, title=title)


if __name__ == "__main__":
    main()
