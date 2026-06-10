"""
End-to-end assembled-physics gate (the test that was missing).

Element-level checks (KE vs lk_H8, FD sensitivities, einsum rotation) are all
blind to a node-ordering mismatch between the element matrix and build_edof:
FD is self-consistent, and the isotropic KE comparison can pass in the wrong
order. The only unambiguous gate is an assembled boundary-value problem with a
known answer, solved through the real SpinodalFEA.analyze path:

  1. isotropic solid bar under uniform axial tension matches the analytic
     compliance c = F_tot^2 L / (E A) to a few percent;
  2. columnar rho=0.3 bar: compliance ratio perpendicular/aligned matches the
     Young's modulus anisotropy E3/E1 (~11.5) from the homogenized data.

Run:  python -m spinodal_pytopo3d._bar_check
"""

from __future__ import annotations

import sys

import numpy as np

from spinodal_pytopo3d import spinodal_material as sm
from spinodal_pytopo3d.spinodal_fea import SpinodalFEA, SpinodalParams


def _bar(nelx=12, nely=4, nelz=4):
    from pytopo3d.utils.assembly import build_supports
    from pytopo3d.utils.filter import build_filter

    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    plane = (nelx + 1) * (nely + 1)
    F = np.zeros(ndof)
    for iy in range(nely + 1):
        for iz in range(nelz + 1):
            nid = iz * plane + nelx * (nely + 1) + iy
            F[3 * nid] = 1.0                      # +x1 on the free end face
    free, _ = build_supports(nelx, nely, nelz, ndof)
    H, Hs = build_filter(nelx, nely, nelz, 1.01)
    fea = SpinodalFEA(nelx, nely, nelz, F, free, H, Hs, spinodal_class="columnar")
    return fea, F, nelx, nely, nelz


def main():
    fea, F, nelx, nely, nelz = _bar()
    params = SpinodalParams(penal=3.0, beta=25.0, Emin=1e-9)
    n = nelx * nely * nelz
    ones, zero = np.ones(n), np.zeros(n)
    ok = True

    # --- 1. isotropic solid bar vs analytic ---------------------------------
    c_iso = fea.analyze(ones, ones, zero, zero, zero, params)["c"]
    ftot = F.sum()
    c_ref = ftot * ftot * nelx / (1.0 * nely * nelz)     # F^2 L / (E A), E=1
    err = abs(c_iso - c_ref) / c_ref
    print(f"isotropic bar: c = {c_iso:.2f} vs analytic {c_ref:.2f}  (err {100*err:.1f}%)")
    if err > 0.05:
        ok = False
        print("  FAIL: assembled isotropic physics does not match the analytic bar")

    # --- 2. columnar anisotropy ratio ----------------------------------------
    frac = np.full(n, 0.3)
    c_perp = fea.analyze(ones, frac, zero, zero, zero, params)["c"]          # axis=x3
    c_alig = fea.analyze(ones, frac, zero, np.full(n, np.pi / 2), zero,
                         params)["c"]                                        # axis=x1
    ratio = c_perp / c_alig
    coeff = sm.load_coeff("columnar")
    D = sm.ch_of_density(np.array([0.3]), coeff)[0][0]
    S = np.linalg.inv(D)
    e_ratio = S[0, 0] / S[2, 2]                          # E3/E1 from compliance
    print(f"columnar rho=0.3 bar: c_perp/c_aligned = {ratio:.2f} "
          f"(Young's E3/E1 = {e_ratio:.2f})")
    if not (0.75 * e_ratio < ratio < 1.25 * e_ratio):
        ok = False
        print("  FAIL: assembled anisotropy does not reproduce the material data")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "PyTopo3D-main"))
    sys.exit(main())
