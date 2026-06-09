"""
Spinodal finite-element analysis + analytic sensitivities (single columnar class),
built on the PyTopo3D mesh/DOF/solver backbone.

Per element e the constitutive matrix is
    D_e = E_simp(z~_e) * N(angles_e)^T  D^H(rho_e)  N(angles_e),
where z~ = filter(z) is the macro presence field (SIMP+Heaviside), rho is the
spinodal density, and N is the 6x6 Voigt rotation. KE_e = sum_g B_g^T D_e B_g.

Objective: compliance c = F^T u = sum_e E_simp_e * ce0_e, with
    ce0_e = sum_g (N eps_g)^T D^H (N eps_g),   eps_g = B_g u_e.

Sensitivities (compliance is self-adjoint, dc/dp = -u^T (dK/dp) u):
    dc/dE_simp_e = -ce0_e                       (-> chain to z via Heaviside+filter)
    dc/drho_e    = -E_simp_e * sum_g (N eps)^T dD^H/drho (N eps)
    dc/dang_e    = -E_simp_e * 2 sum_g (dN eps)^T D^H (N eps)

Volume measure: V_e = rho_bar_e * rho_e (presence x spinodal density),
vol = mean_e V_e ; constraint g = vol - volfrac.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "PyTopo3D-main"))
from pytopo3d.utils.assembly import build_edof  # noqa: E402

from . import spinodal_material as sm
from .fea_element import element_B_integrated
from .interpolation import macro_interpolation


def _make_scatter_map(i_full, j_full, ndof):
    linear = i_full.astype(np.int64) * ndof + j_full
    uniq_lin, dup2uniq = np.unique(linear, return_inverse=True)
    return uniq_lin // ndof, uniq_lin % ndof, dup2uniq


@dataclass
class SpinodalParams:
    penal: float = 3.0
    beta: float = 1.0          # Heaviside sharpness
    eta: float = 0.5
    Emin: float = 1e-9


class SpinodalFEA:
    """Assembles, solves and differentiates the single-class spinodal model."""

    def __init__(self, nelx, nely, nelz, F, freedofs0, H, Hs,
                 spinodal_class="columnar", mat_path=None, h=1.0, use_pardiso=True,
                 use_gpu=False, cg_tol=1e-8):
        self.nelx, self.nely, self.nelz = nelx, nely, nelz
        self.nele = nelx * nely * nelz
        self.ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
        self.F = F
        self.freedofs0 = freedofs0
        self.H, self.Hs = H, Hs

        self.B_int = element_B_integrated(h)          # (8,6,24)
        self.coeff = sm.load_coeff(spinodal_class, mat_path)

        edofMat, _, _ = build_edof(nelx, nely, nelz)
        self.edof0 = edofMat - 1                       # 0-based for U indexing

        # (1) GEMM assembly tensor: per element KE_flat(576) = D_flat(36) @ Mmat,
        #     replacing the Gauss-loop einsum + large non-contiguous ravel.
        M4 = np.einsum("gsi,gtj->stij", self.B_int, self.B_int, optimize=True)
        self.Mmat = np.ascontiguousarray(M4.reshape(36, 576))

        # (2) assemble the reduced (free-dof) stiffness directly, fixed pattern.
        #     Per-element 576 entries are ordered C-style k = i*24 + j (matches Mmat cols).
        iK0 = np.repeat(self.edof0, 24, axis=1).ravel()   # row dof (i outer)
        jK0 = np.tile(self.edof0, (1, 24)).ravel()        # col dof (j inner)
        nfree = freedofs0.size
        g2r = np.full(self.ndof, -1, dtype=np.int64)
        g2r[freedofs0] = np.arange(nfree)
        ri, rj = g2r[iK0], g2r[jK0]
        self._active = (ri >= 0) & (rj >= 0)              # keep free-free couplings
        i_u, j_u, self._dup_r = _make_scatter_map(ri[self._active], rj[self._active], nfree)
        self.Kff = sp.csr_matrix((np.zeros(len(i_u)), (i_u, j_u)), shape=(nfree, nfree))
        self.Kff.sort_indices()
        self._nfree = nfree

        # Optional GPU solve (CG + Jacobi on the RTX). Assembly stays on CPU; only
        # the reduced linear system goes to the GPU (persistent GPU matrix, values
        # refreshed each iteration). Wins on large meshes; for ~30k dof a CPU direct
        # solver is usually competitive or faster -- benchmark before committing.
        self._gpu = False
        self._cg_tol = cg_tol
        if use_gpu:
            try:
                self._setup_gpu()
                self._gpu = True
                self.solver_name = "CuPy CG + Jacobi (GPU)"
            except Exception as exc:
                print(f"[gpu] unavailable, falling back to CPU ({exc})")

        # CPU solver (also the fallback). Fresh pypardiso.spsolve per call is,
        # empirically, faster here than a persistent PyPardisoSolver.
        self._solver = None
        if use_pardiso:
            try:
                import pypardiso
                self._solver = pypardiso.spsolve
                if not self._gpu:
                    self.solver_name = "PyPardiso (multi-core)"
            except Exception:
                self._solver = None
        if self._solver is None:
            from scipy.sparse.linalg import spsolve
            self._solver = spsolve
            if not self._gpu:
                self.solver_name = "SciPy spsolve"

    def _setup_gpu(self):
        import inspect

        import cupy as cp
        import cupyx.scipy.sparse as cusp
        from cupyx.scipy.sparse.linalg import LinearOperator, cg
        self._cp, self._cusp, self._cg, self._LO = cp, cusp, cg, LinearOperator
        # scipy/cupy renamed the CG tolerance kwarg tol -> rtol; pick what exists.
        self._tol_kw = "rtol" if "rtol" in inspect.signature(cg).parameters else "tol"
        self._gKff = cusp.csr_matrix(self.Kff)        # structure once
        self._x_prev = None                            # CG warm-start state
        # force one tiny solve to compile kernels & validate the CUDA stack
        _ = cg(self._gKff + cusp.identity(self._nfree, format="csr"),
               cp.ones(self._nfree), maxiter=1)

    def _solve_free_gpu(self, b):
        cp = self._cp
        self._gKff.data = cp.asarray(self.Kff.data)   # refresh values only
        m_inv = 1.0 / (self._gKff.diagonal() + 1e-20)
        M = self._LO(self._gKff.shape, matvec=lambda x: m_inv * x)
        kw = {self._tol_kw: self._cg_tol, "maxiter": 20000, "M": M}
        # warm-start from the previous solution: in the optimization loop the
        # design (and hence the solution) changes slowly, so this cuts CG
        # iterations dramatically. Converges to the same answer within tol.
        x, info = self._cg(self._gKff, cp.asarray(b), x0=self._x_prev, **kw)
        if info != 0:
            raise RuntimeError(f"GPU CG did not converge (info={info})")
        self._x_prev = x
        return cp.asnumpy(x)

    def _solve_free(self, b):
        if self._gpu:
            return self._solve_free_gpu(b)
        return self._solver(self.Kff, b)

    def macro_material_cache(self, z, Frac, params: SpinodalParams):
        """Precompute the z/Frac-dependent fields (frozen during angle sub-iters)."""
        rho_tilde = self.filter_fwd(z)
        E_simp, dE_drt, rho_bar, d_rho_bar = macro_interpolation(
            rho_tilde, params.penal, params.beta, params.eta, params.Emin
        )
        CH, dCH = sm.ch_of_density(Frac, self.coeff)
        return (E_simp, dE_drt, rho_bar, d_rho_bar, CH, dCH)

    # ---- forward filtering -------------------------------------------------
    def filter_fwd(self, z):
        return (self.H @ z) / self.Hs

    def filter_T(self, v):
        # H is symmetric; chain rule for rho~ = H z / Hs is H^T (v / Hs)
        return self.H.T @ (v / self.Hs)

    # ---- main routine -----------------------------------------------------
    def analyze(self, z, Frac, alpha, beta, gamma, params: SpinodalParams,
                cache=None) -> Dict:
        nele = self.nele
        # (4) reuse frozen z/Frac-dependent fields when provided (angle sub-iters)
        if cache is None:
            cache = self.macro_material_cache(z, Frac, params)
        E_simp, dE_drt, rho_bar, d_rho_bar, CH, dCH = cache

        N, dNa, dNb, dNg = sm.rotation_voigt(alpha, beta, gamma)      # (nele,6,6)

        # rotated homogenized tensor  N^T CH N  (before macro SIMP scaling)
        NchN = np.einsum("eki,ekl,elj->eij", N, CH, N, optimize=True)
        D_all = E_simp[:, None, None] * NchN

        # (1) per-element 24x24 via a single GEMM, then (2) scatter directly into Kff
        elem_vals = (D_all.reshape(nele, 36) @ self.Mmat).ravel()     # (nele*576,)
        self.Kff.data[:] = 0.0
        np.add.at(self.Kff.data, self._dup_r, elem_vals[self._active])
        fd = self.freedofs0
        U = np.zeros(self.ndof)
        U[fd] = self._solve_free(self.F[fd])

        # element strains and rotated strains
        Ue = U[self.edof0]                                           # (nele,24)
        eps = np.einsum("gsd,ed->egs", self.B_int, Ue, optimize=True)   # (nele,8,6)
        Neps = np.einsum("eik,egk->egi", N, eps, optimize=True)         # N eps

        ce0 = np.einsum("egi,eij,egj->e", Neps, CH, Neps, optimize=True)   # u^T KE0 u
        c = float(np.sum(E_simp * ce0))

        # sensitivities
        dc_dE = -ce0
        dc_dz = self.filter_T(dc_dE * dE_drt)

        dc_dFrac = -E_simp * np.einsum("egi,eij,egj->e", Neps, dCH, Neps, optimize=True)

        def ang_sens(dN):
            dNeps = np.einsum("eik,egk->egi", dN, eps, optimize=True)
            return -E_simp * 2.0 * np.einsum("egi,eij,egj->e", dNeps, CH, Neps, optimize=True)

        dc_da, dc_db, dc_dg = ang_sens(dNa), ang_sens(dNb), ang_sens(dNg)

        # volume constraint (presence x spinodal density)
        V = rho_bar * Frac
        vol = float(np.mean(V))
        dvol_dz = self.filter_T(d_rho_bar * Frac / nele)
        dvol_dFrac = rho_bar / nele

        return dict(
            U=U, c=c, vol=vol, rho_bar=rho_bar, E_simp=E_simp, V=V,
            dc_dz=dc_dz, dc_dFrac=dc_dFrac, dc_da=dc_da, dc_db=dc_db, dc_dg=dc_dg,
            dvol_dz=dvol_dz, dvol_dFrac=dvol_dFrac,
        )
