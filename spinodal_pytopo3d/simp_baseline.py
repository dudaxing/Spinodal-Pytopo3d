"""
Self-contained standard solid-isotropic SIMP + OC compliance minimization.

Used to obtain the baseline compliance f0 that Senhora 2022 normalizes against
(f/f0). Deliberately independent of PyTopo3D's `top3d`/`oc_update`/`apply_filter`,
whose CPU paths reference CuPy symbols unconditionally and crash without a GPU.
Reuses only PyTopo3D's bug-free helpers (mesh/edof/element/filter/solver) so the
FEA is identical to the spinodal model's (verified: our B == lk_H8).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "PyTopo3D-main"))
from pytopo3d.utils.assembly import build_edof  # noqa: E402
from pytopo3d.utils.stiffness import lk_H8  # noqa: E402

from spinodal_pytopo3d.spinodal_fea import _make_scatter_map


def simp_topopt(nelx, nely, nelz, volfrac, penal, H, Hs, F, freedofs0,
                maxiter=300, tol=1e-3, move=0.2, Emin=1e-9, E0=1.0, nu=0.3,
                use_pardiso=True, verbose=True):
    """Standard SIMP+OC. Returns (xPhys, compliance)."""
    nele = nelx * nely * nelz
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    KE = lk_H8(nu)
    KE_flat = KE.ravel()
    edofMat, _, _ = build_edof(nelx, nely, nelz)
    edof0 = edofMat - 1

    # assemble the reduced (free-dof) stiffness directly, fixed pattern (no per-iter slicing)
    iK0 = np.repeat(edof0, 24, axis=1).ravel()
    jK0 = np.tile(edof0, (1, 24)).ravel()
    nfree = freedofs0.size
    g2r = np.full(ndof, -1, dtype=np.int64)
    g2r[freedofs0] = np.arange(nfree)
    ri, rj = g2r[iK0], g2r[jK0]
    active = (ri >= 0) & (rj >= 0)
    i_u, j_u, dup_r = _make_scatter_map(ri[active], rj[active], nfree)
    Kff = sp.csr_matrix((np.zeros(len(i_u)), (i_u, j_u)), shape=(nfree, nfree))
    Kff.sort_indices()

    if use_pardiso:
        try:
            import pypardiso
            solver, name = pypardiso.spsolve, "PyPardiso (multi-core)"
        except Exception:
            from scipy.sparse.linalg import spsolve
            solver, name = spsolve, "SciPy spsolve"
    else:
        from scipy.sparse.linalg import spsolve
        solver, name = spsolve, "SciPy spsolve"
    if verbose:
        print(f"[baseline] solver: {name}")

    x = np.full(nele, volfrac)
    xPhys = (H @ x) / Hs
    c = np.inf

    for it in range(maxiter):
        stiff = Emin + xPhys ** penal * (E0 - Emin)
        elem_vals = np.kron(stiff, KE_flat)
        Kff.data[:] = 0.0
        np.add.at(Kff.data, dup_r, elem_vals[active])
        U = np.zeros(ndof)
        U[freedofs0] = solver(Kff, F[freedofs0])

        Ue = U[edof0]
        ce = np.einsum("ij,jk,ik->i", Ue, KE, Ue, optimize=True)
        c = float(np.sum(stiff * ce))
        dc = -penal * (E0 - Emin) * xPhys ** (penal - 1) * ce
        dv = np.ones(nele)
        dc = H @ (dc / Hs)
        dv = H @ (dv / Hs)

        # OC bisection
        l1, l2 = 1e-9, 1e9
        xnew = x
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l1 + l2)
            xnew = np.maximum(0.0, np.maximum(x - move,
                    np.minimum(1.0, np.minimum(x + move,
                    x * np.sqrt(np.maximum(-dc / dv / lmid, 0.0))))))
            xPhys = (H @ xnew) / Hs
            if xPhys.mean() > volfrac:
                l1 = lmid
            else:
                l2 = lmid
        change = float(np.max(np.abs(xnew - x)))
        x = xnew

        if verbose and (it % 20 == 0 or it == maxiter - 1):
            print(f"[baseline] it {it:4d} | c={c:.4e} | vol={xPhys.mean():.4f} | chg={change:.3f}")
        if change < tol:
            if verbose:
                print(f"[baseline] converged at it {it} (change {change:.2e})")
            break

    return xPhys, c
