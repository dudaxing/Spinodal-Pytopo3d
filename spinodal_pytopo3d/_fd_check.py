"""Finite-difference verification of SpinodalFEA analytic sensitivities.

Run from project root:  python -m spinodal_pytopo3d._fd_check
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "PyTopo3D-main"))
from pytopo3d.utils.assembly import build_force_vector, build_supports  # noqa: E402
from pytopo3d.utils.filter import build_filter  # noqa: E402

from spinodal_pytopo3d.spinodal_fea import SpinodalFEA, SpinodalParams


def main():
    rng = np.random.default_rng(0)
    nelx, nely, nelz = 4, 3, 2
    nele = nelx * nely * nelz
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)

    F = build_force_vector(nelx, nely, nelz, ndof)
    freedofs0, _ = build_supports(nelx, nely, nelz, ndof)
    H, Hs = build_filter(nelx, nely, nelz, rmin=1.5)

    fea = SpinodalFEA(nelx, nely, nelz, F, freedofs0, H, Hs,
                      spinodal_class="columnar", use_pardiso=False)
    p = SpinodalParams(penal=3.0, beta=2.0, eta=0.5, Emin=1e-9)

    z = rng.uniform(0.4, 0.8, nele)
    Frac = rng.uniform(0.3, 0.7, nele)
    a = rng.uniform(-1.0, 1.0, nele)
    b = rng.uniform(-1.0, 1.0, nele)
    g = rng.uniform(-1.0, 1.0, nele)

    out = fea.analyze(z, Frac, a, b, g, p)
    c0 = out["c"]
    # consistency: c == F^T U
    print(f"c                 = {c0:.8e}")
    print(f"F^T U             = {float(F @ out['U']):.8e}  (should equal c)")

    h = 1e-6

    def c_of(zz, ff, aa, bb, gg):
        return fea.analyze(zz, ff, aa, bb, gg, p)["c"]

    def check(name, vec, grad, builder):
        idx = [0, nele // 2, nele - 1]
        errs = []
        for i in idx:
            vp = vec.copy(); vp[i] += h
            vm = vec.copy(); vm[i] -= h
            fd = (builder(vp) - builder(vm)) / (2 * h)
            an = grad[i]
            denom = max(abs(an), abs(fd), 1e-12)
            errs.append(abs(fd - an) / denom)
        print(f"{name:10s} rel-err @3 elems: " + ", ".join(f"{e:.2e}" for e in errs))
        return max(errs)

    m = []
    m.append(check("dc/dz", z, out["dc_dz"], lambda v: c_of(v, Frac, a, b, g)))
    m.append(check("dc/dFrac", Frac, out["dc_dFrac"], lambda v: c_of(z, v, a, b, g)))
    m.append(check("dc/dalpha", a, out["dc_da"], lambda v: c_of(z, Frac, v, b, g)))
    m.append(check("dc/dbeta", b, out["dc_db"], lambda v: c_of(z, Frac, a, v, g)))
    m.append(check("dc/dgamma", g, out["dc_dg"], lambda v: c_of(z, Frac, a, b, v)))

    # volume sensitivities
    def vol_of(zz, ff):
        return fea.analyze(zz, ff, a, b, g, p)["vol"]

    m.append(check("dvol/dz", z, out["dvol_dz"], lambda v: vol_of(v, Frac)))
    m.append(check("dvol/dFrac", Frac, out["dvol_dFrac"], lambda v: vol_of(z, v)))

    worst = max(m)
    print(f"\nWORST relative error = {worst:.2e}")
    print("PASS" if worst < 1e-4 else "FAIL")


if __name__ == "__main__":
    main()
