"""
Half-domain symmetry gate: the `--symmetry half-y` model must be *exactly*
equivalent to the full-domain model for a symmetric structure.

A full domain (nelx x 2n x nelz) with the centered nodal tip load F and a
solid design is mirror-symmetric about the beam center plane, so the half
model (nelx x n x nelz, u_y = 0 on y=0, load F/2 on the plane) must produce
the identical displacement at the load node, to solver precision. This check
is independent of the density filter (uniform z), so it isolates the BC/load
bookkeeping of `build_cantilever_supports` / `tip_load_force`.

Run:  python -m spinodal_pytopo3d._sym_check
"""

from __future__ import annotations

import sys

import numpy as np

from spinodal_pytopo3d import run_cantilever as rc
from spinodal_pytopo3d.spinodal_fea import SpinodalFEA, SpinodalParams


def _solve(nelx, nely, nelz, sym, fmag, frac):
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    F = rc.tip_load_force(nelx, nely, nelz, ndof, sym) * fmag
    free, _ = rc.build_cantilever_supports(nelx, nely, nelz, ndof, sym)
    H, Hs = rc.build_filter(nelx, nely, nelz, nelx / 36.0)
    fea = SpinodalFEA(nelx, nely, nelz, F, free, H, Hs, spinodal_class="columnar")
    n = nelx * nely * nelz
    params = SpinodalParams(penal=3.0, beta=25.0, Emin=1e-4)
    out = fea.analyze(np.ones(n), np.full(n, frac), np.zeros(n), np.zeros(n),
                      np.zeros(n), params)
    ldof = int(np.flatnonzero(np.abs(F) > 0)[0])
    return out["U"][ldof]


def main():
    ok = True
    for frac, label in ((1.0, "isotropic solid"), (0.5, "columnar rho=0.5")):
        u_full = _solve(36, 12, 12, "none", 1.0, frac)
        u_half = _solve(36, 6, 12, "half-y", 0.5, frac)
        rel = abs(u_full - u_half) / abs(u_full)
        print(f"{label}: u_full={u_full:.8f} u_half={u_half:.8f} rel.diff={rel:.2e}")
        if rel > 1e-8:
            ok = False
            print("  FAIL: half-y model is not equivalent to the full domain")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
