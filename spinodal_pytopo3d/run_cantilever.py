"""
Reproduce the spinodal cantilever of Senhora, Sanders & Paulino (Adv. Mater. 2022),
Fig. 5, on the PyTopo3D backbone (single columnar class, CPU).

Pipeline:
  1. standard solid-isotropic SIMP baseline  -> compliance f0  (via PyTopo3D top3d)
  2. spinodal compliance minimization        -> compliance f
        * "shell": spinodal density fixed at rho_min (=0.3)
        * "truss": spinodal density free in [0.3, 0.7]
  3. report f/f0  (paper: truss < 1 beats solid; shell > 1)

Run from project root, e.g.:
  python -m spinodal_pytopo3d.run_cantilever --nelx 32 --nely 16 --nelz 16 \
         --volfrac 0.05 --rmin 1.5 --maxiter 300 --case both
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "PyTopo3D-main"))
from pytopo3d.utils.assembly import build_force_vector, build_supports  # noqa: E402
from pytopo3d.utils.filter import build_filter  # noqa: E402

from spinodal_pytopo3d.spinodal_fea import SpinodalFEA
from spinodal_pytopo3d.simp_baseline import simp_topopt
from spinodal_pytopo3d.optimizer import OptOptions, optimize

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def node_id0(ix, iy, iz, nelx, nely, nelz):
    """0-based node id for PyTopo3D's Fortran-order node numbering."""
    if not (0 <= ix <= nelx and 0 <= iy <= nely and 0 <= iz <= nelz):
        raise ValueError(f"node ({ix}, {iy}, {iz}) outside grid {nelx}x{nely}x{nelz}")
    return iy + ix * (nely + 1) + iz * (nelx + 1) * (nely + 1)


def tip_load_force(nelx, nely, nelz, ndof, symmetry="none"):
    """Concentrated -x3 nodal load at the center of the free-end face.

    The earlier implementation used a force_field entry on the last element and
    PyTopo3D distributed it to that element's eight corner nodes. Fig. 5's load
    is a point load on the right face center, so apply it directly to the node
    (x=nelx, y=nely/2, z=nelz/2).
    """
    F = np.zeros(ndof)
    ix = nelx
    iy = 0 if symmetry == "half-y" else nely // 2
    iz = nelz // 2
    nid = node_id0(ix, iy, iz, nelx, nely, nelz)
    F[3 * nid + 2] = -1.0
    return F


def load_summary(F, nelx, nely, nelz):
    """Return loaded nodal coordinates and force components for diagnostics."""
    out = []
    for dof in np.flatnonzero(np.abs(F) > 0):
        nid, comp = divmod(int(dof), 3)
        plane = (nelx + 1) * (nely + 1)
        iz = nid // plane
        rem = nid - iz * plane
        ix = rem // (nely + 1)
        iy = rem - ix * (nely + 1)
        out.append((ix, iy, iz, comp, float(F[dof])))
    return out


def make_load_pad_mask(nelx, nely, nelz, radius, symmetry="none"):
    """Element mask for a small passive pad around the free-end center load node."""
    mask = np.zeros(nelx * nely * nelz, dtype=bool)
    if radius <= 0:
        return mask
    cx = float(nelx)
    cy = 0.0 if symmetry == "half-y" else float(nely) / 2.0
    cz = float(nelz) / 2.0
    r2 = float(radius) ** 2
    for elz in range(nelz):
        for elx in range(nelx):
            for ely in range(nely):
                x, y, z = elx + 0.5, ely + 0.5, elz + 0.5
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r2:
                    mask[ely + elx * nely + elz * nelx * nely] = True
    return mask


def build_cantilever_supports(nelx, nely, nelz, ndof, symmetry="none"):
    """Left-face clamp plus optional half-domain symmetry constraints.

    `half-y` represents the paper's half-width computational domain. The
    symmetry plane is y=0, parallel to x1-x3, so only the normal displacement
    component u_y is fixed there. The left face remains fully clamped.
    """
    freedofs0, fixeddof0 = build_supports(nelx, nely, nelz, ndof)
    if symmetry == "none":
        return freedofs0, fixeddof0
    if symmetry != "half-y":
        raise ValueError(f"unsupported symmetry mode: {symmetry}")

    sym_nodes = []
    for iz in range(nelz + 1):
        for ix in range(nelx + 1):
            sym_nodes.append(node_id0(ix, 0, iz, nelx, nely, nelz))
    sym_dofs = 3 * np.asarray(sym_nodes, dtype=int) + 1
    fixeddof0 = np.unique(np.concatenate([fixeddof0, sym_dofs]))
    all_dofs0 = np.arange(ndof)
    freedofs0 = all_dofs0[~np.isin(all_dofs0, fixeddof0, assume_unique=True)]
    return freedofs0, fixeddof0


def run(args):
    os.makedirs(RESULTS, exist_ok=True)
    nelx, nely, nelz = args.nelx, args.nely, args.nelz
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)

    rmin, Emin = args.rmin, 1e-9
    if args.fig5:
        args.load = "tip"
        rmin = nelx / 36.0            # R=0.4cm on a 14.4cm (=nelx el) beam
        Emin = 1e-4
        print(f"[fig5] faithful: 3:1:1 domain, tip load, rmin={rmin:.2f} (R=0.4cm), Ersatz={Emin}")
    if args.symmetry == "half-y":
        print("[symmetry] half-y: solve half-width domain with u_y=0 on the x1-x3 center plane")

    F = (tip_load_force(nelx, nely, nelz, ndof, args.symmetry) if args.load == "tip"
         else build_force_vector(nelx, nely, nelz, ndof))
    load_info = load_summary(F, nelx, nely, nelz)
    print(f"[load] nonzero nodal loads: {load_info}")
    freedofs0, fixeddof0 = build_cantilever_supports(nelx, nely, nelz, ndof, args.symmetry)
    print(f"[support] fixed DOFs: {fixeddof0.size}")
    passive_z = make_load_pad_mask(nelx, nely, nelz, args.load_pad_radius, args.symmetry)
    passive_frac_value = args.load_pad_frac if passive_z.any() else None
    if passive_z.any():
        pad_vol = passive_z.sum() * passive_frac_value / (nelx * nely * nelz)
        print(f"[load-pad] passive elements={int(passive_z.sum())}, radius={args.load_pad_radius:g}, "
              f"Frac={passive_frac_value:g}, minimum volume contribution={pad_vol:.5f}")
    H, Hs = build_filter(nelx, nely, nelz, rmin)

    fea = SpinodalFEA(nelx, nely, nelz, F, freedofs0, H, Hs,
                      spinodal_class="columnar", use_pardiso=not args.no_pardiso,
                      use_gpu=args.use_gpu)
    print(f"[solver] {fea.solver_name}")

    # ---- baseline f0 (skippable) ----
    if args.no_baseline:
        f0 = float("nan")
        print("[baseline] skipped (--no-baseline); f/f0 will be NaN")
    elif args.fig5:
        # CONSISTENT baseline: solid isotropic via the SAME optimizer. Columnar Dᴴ at
        # rho=1 is the isotropic solid base, so f0 and the spinodal c share identical
        # continuation/convergence -> fair f/f0 (basic-OC simp_topopt under-converges).
        print("=" * 60, "\n[baseline] solid-isotropic via spinodal optimizer (rho=1)\n", "=" * 60, sep="")
        opt0 = OptOptions(
            volfrac=args.volfrac, rho_min=1.0, rho_max=1.0, optimize_density=False,
            move_z=0.05, move_frac=0.0, move_angle=0.0, Emin=Emin, angle_subiters=0,
            penal_steps=(1.0, 1.5, 2.0, 2.5, 3.0), penal_iters=(150, 100, 100, 50, 50),
            beta0=1.0, beta_add=0.5, beta_period=15, beta_max=25.0,
            passive_z=passive_z, passive_frac_value=1.0 if passive_z.any() else None,
            max_iter=args.maxiter, verbose=False,
        )
        f0 = optimize(fea, opt0)["c"]
        print(f"\n[baseline] f0 = {f0:.6e}\n")
    else:
        print("=" * 60, "\n[baseline] standard solid-isotropic SIMP\n", "=" * 60, sep="")
        xPhys0, f0 = simp_topopt(nelx, nely, nelz, args.volfrac, args.penal, H, Hs,
                                 F, freedofs0, maxiter=max(args.maxiter, 300),
                                 use_pardiso=not args.no_pardiso, Emin=Emin)
        print(f"\n[baseline] f0 = {f0:.6e}\n")

    cases = ["shell", "truss"] if args.case == "both" else [args.case]
    summary = {"f0": f0}
    for case in cases:
        print("=" * 60, f"\n[spinodal] case = {case}\n", "=" * 60, sep="")
        if args.fig5:
            opt = OptOptions(
                volfrac=args.volfrac, rho_min=0.3, rho_max=0.7,
                optimize_density=(case == "truss"),
                move_z=0.05, move_frac=0.05, move_angle=0.25, Emin=Emin,
                penal_steps=(1.0, 1.5, 2.0, 2.5, 3.0), penal_iters=(150, 100, 100, 50, 50),
                beta0=1.0, beta_add=0.5, beta_period=15, beta_max=25.0,
                angle_subiters=30, angle_period=25, angle_phase_iter=150,
                passive_z=passive_z, passive_frac_value=passive_frac_value,
                max_iter=args.maxiter,
            )
        else:
            opt = OptOptions(
                volfrac=args.volfrac, penal=args.penal,
                rho_min=0.3, rho_max=0.7,
                optimize_density=(case == "truss"),
                passive_z=passive_z, passive_frac_value=passive_frac_value,
                max_iter=args.maxiter,
                beta_start_iter=min(150, args.maxiter // 3),
            )
        res = optimize(fea, opt)
        f = res["c"]
        ratio = f / f0
        verdict = "beats solid (f/f0<1)" if ratio < 1 else "porous trade-off (f/f0>1)"
        frac_summary = _frac_summary(res["Frac"])
        print(f"\n[spinodal:{case}] f = {f:.6e} | f/f0 = {ratio:.4f}  -> {verdict}")
        print(f"[spinodal:{case}] Frac { _format_frac_summary(frac_summary) }")
        summary[case] = {"f": f, "ratio": ratio,
                         "mean_V": float(np.mean(res["V"])),
                         "frac_summary": frac_summary}

        label = f"{case}{args.tag}"
        np.savez(
            os.path.join(RESULTS, f"cantilever_{label}.npz"),
            z=res["z"], Frac=res["Frac"], alpha=res["alpha"], beta=res["beta"],
            gamma=res["gamma"], rho_bar=res["rho_bar"], V=res["V"],
            c=f, f0=f0, ratio=ratio,
            history_c=np.array(res["history"]["c"]),
            history_vol=np.array(res["history"]["vol"]),
            history_g=np.array(res["history"]["g"]),
            history_change=np.array(res["history"]["change"]),
            history_beta=np.array(res["history"]["beta"]),
            history_penal=np.array(res["history"]["penal"]),
            frac_min=frac_summary["min"], frac_max=frac_summary["max"],
            frac_p10=frac_summary["p10"], frac_p50=frac_summary["p50"],
            frac_p90=frac_summary["p90"], frac_hi065=frac_summary["hi065"],
            frac_lo035=frac_summary["lo035"],
            load_info=np.array(load_info, dtype=float),
            fixed_dof_count=int(fixeddof0.size),
            passive_z=passive_z.astype(np.uint8),
            load_pad_radius=float(args.load_pad_radius),
            load_pad_frac=float(passive_frac_value) if passive_frac_value is not None else np.nan,
            symmetry=np.array(args.symmetry),
            nelx=nelx, nely=nely, nelz=nelz, volfrac=args.volfrac,
        )
        _save_plots(res, label, f0)
        npz = os.path.join(RESULTS, f"cantilever_{label}.npz")
        try:
            from spinodal_pytopo3d.visualize import visualize
            visualize(npz, quiver_stride=max(1, (nelx * nely * nelz) // 1500))
        except Exception as exc:  # visualization is best-effort
            print(f"[viz] skipped ({exc})")
        if args.stl or args.vtk:
            from spinodal_pytopo3d import export as _ex
            if args.stl:
                try:
                    _ex.export_stl(npz)
                except Exception as exc:
                    print(f"[stl] skipped ({exc})")
            if args.vtk:
                _ex.export_vtk(npz)

    print("\n" + "=" * 60, "\nSUMMARY\n", "=" * 60, sep="")
    print(f"f0 (solid SIMP)         = {f0:.6e}")
    for case in cases:
        s = summary[case]
        print(f"{case:6s}: f = {s['f']:.6e}  f/f0 = {s['ratio']:.4f}  "
              f"meanV = {s['mean_V']:.4f}  Frac { _format_frac_summary(s['frac_summary']) }")
    return summary


def _frac_summary(frac):
    frac = np.asarray(frac, dtype=float).ravel()
    p10, p50, p90 = np.percentile(frac, [10, 50, 90])
    return {
        "min": float(np.min(frac)),
        "max": float(np.max(frac)),
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "hi065": float(np.mean(frac >= 0.65)),
        "lo035": float(np.mean(frac <= 0.35)),
    }


def _format_frac_summary(s):
    return (f"min={s['min']:.3f} p10={s['p10']:.3f} p50={s['p50']:.3f} "
            f"p90={s['p90']:.3f} max={s['max']:.3f} "
            f"hi>=0.65={100*s['hi065']:.1f}% lo<=0.35={100*s['lo035']:.1f}%")


def _save_plots(res, case, f0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = res["history"]
    c = np.array(h["c"]); vol = np.array(h["vol"])
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(c / f0, "b-", label="f/f0")
    ax1.set_xlabel("iteration"); ax1.set_ylabel("f / f0", color="b")
    ax1.axhline(1.0, color="b", ls=":", lw=0.8)
    ax2 = ax1.twinx()
    ax2.plot(vol, "r-", label="vol")
    ax2.set_ylabel("volume fraction", color="r")
    ax1.set_title(f"Spinodal cantilever ({case}): convergence")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f"cantilever_{case}_convergence.png"), dpi=130)
    plt.close(fig)


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nelx", type=int, default=32)
    p.add_argument("--nely", type=int, default=16)
    p.add_argument("--nelz", type=int, default=16)
    p.add_argument("--volfrac", type=float, default=0.05)
    p.add_argument("--penal", type=float, default=3.0)
    p.add_argument("--rmin", type=float, default=1.5)
    p.add_argument("--maxiter", type=int, default=300)
    p.add_argument("--case", choices=["shell", "truss", "both"], default="both")
    p.add_argument("--tag", default="", help="suffix for output names (e.g. _64 to avoid overwrite)")
    p.add_argument("--load", choices=["edge", "tip"], default="edge",
                   help="edge=distributed free-end edge (default); tip=nodal point load at free-end center")
    p.add_argument("--load-pad-radius", type=float, default=0.0,
                   help="passive macro pad radius around the free-end center load node, in elements")
    p.add_argument("--load-pad-frac", type=float, default=0.7,
                   help="spinodal solid fraction pinned inside the passive load pad")
    p.add_argument("--symmetry", choices=["none", "half-y"], default="none",
                   help="half-y solves the half-width Fig.5 domain with u_y=0 on y=0")
    p.add_argument("--fig5", action="store_true",
                   help="faithful Adv.Mater.2022 Fig.5 setup (tip load, R=0.4cm, paper "
                        "p-continuation, additive beta->25, Ersatz 1e-4)")
    p.add_argument("--no-pardiso", action="store_true")
    p.add_argument("--no-baseline", action="store_true",
                   help="skip the solid-SIMP baseline (f/f0=NaN); saves time on large meshes")
    p.add_argument("--use-gpu", action="store_true", help="GPU CG solve (cupy/RTX)")
    p.add_argument("--stl", action="store_true", help="export STL of each design")
    p.add_argument("--vtk", action="store_true", help="export VTK (density + orientation)")
    return p


if __name__ == "__main__":
    run(build_argparser().parse_args())
