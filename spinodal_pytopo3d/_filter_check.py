"""Correctness gate for fast_filter.build_filter_fast.

The unambiguous reference is the brute-force triple loop (the MATLAB top3d
filter semantics, kept as a comment in pytopo3d.utils.filter). PyTopo3D's
KD-tree `build_filter` is compared too, but only as information: it builds the
KD-tree on centers raveled with z fastest while the integer-distance pruning
uses the y-fastest element order, so its candidate sets are only correct when
nely == nelz (the index mismatch is then a y<->z transposition, an isometry).
For nely != nelz (e.g. the paper's half-domain 180x30x60) it silently drops
true neighbors. All meshes used for results in this repo so far had
nely == nelz, so they are unaffected.

Run:  python -m spinodal_pytopo3d._filter_check
"""

from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "PyTopo3D-main"))
from pytopo3d.utils.filter import build_filter  # noqa: E402

from spinodal_pytopo3d.fast_filter import build_filter_fast


def build_filter_slow(nelx, nely, nelz, rmin):
    """Brute-force reference (top3d semantics, y-fastest element order)."""
    nele = nelx * nely * nelz
    rows, cols, vals = [], [], []
    for k1 in range(nelz):
        for i1 in range(nelx):
            for j1 in range(nely):
                e1 = k1 * nelx * nely + i1 * nely + j1
                for k2 in range(nelz):
                    for i2 in range(nelx):
                        for j2 in range(nely):
                            d = np.sqrt(max((i1 - i2) ** 2 + (j1 - j2) ** 2
                                            + (k1 - k2) ** 2, 1e-12))
                            w = rmin - d
                            if w > 1e-9:
                                rows.append(e1)
                                cols.append(k2 * nelx * nely + i2 * nely + j2)
                                vals.append(w)
    H = sp.csr_matrix((vals, (rows, cols)), shape=(nele, nele))
    Hs = np.asarray(H.sum(axis=1)).ravel()
    return H, Hs


def main():
    ok = True
    for nelx, nely, nelz, rmin in ((12, 6, 5, 1.5), (9, 7, 6, 2.6),
                                   (8, 4, 4, 8 / 36.0), (10, 5, 4, 3.0),
                                   (6, 6, 6, 2.2)):
        H_ref, Hs_ref = build_filter_slow(nelx, nely, nelz, rmin)
        H_new, Hs_new = build_filter_fast(nelx, nely, nelz, rmin)
        H_up, _ = build_filter(nelx, nely, nelz, rmin)
        dH = abs(H_ref - H_new).max()
        dHs = float(np.max(np.abs(Hs_ref - Hs_new)))
        d_up = abs(H_ref - H_up).max()
        sym = "==" if d_up < 1e-12 else "!="
        print(f"{nelx}x{nely}x{nelz} rmin={rmin:.2f}: nnz={H_new.nnz}  "
              f"max|dH|={dH:.2e}  max|dHs|={dHs:.2e}  "
              f"(upstream {sym} reference, ny==nz: {nely == nelz})")
        if dH > 1e-12 or dHs > 1e-12:
            ok = False
            print("  FAIL: fast filter deviates from the brute-force reference")
        if (nely == nelz) and d_up > 1e-12:
            ok = False
            print("  FAIL: upstream unexpectedly deviates on an ny==nz mesh")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
