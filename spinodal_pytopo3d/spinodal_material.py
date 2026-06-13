"""
Spinodal homogenized material model (single columnar class) + Voigt rotation.

Reuses the homogenization data shipped with the original TO_Spinodal code
(`materials/coefficients_3d.mat`), which stores, for each spinodal class, a
6x6xP array of polynomial coefficients giving the homogenized constitutive
matrix D^H as a function of spinodal density rho:

    D^H(rho) = sum_i  CH_coeff[:, :, i] * rho**(P-1-i)          (P-1 order poly)

Conventions:
  * Voigt strain order [xx, yy, zz, yz, xz, xy] with engineering shear
    (identical to the original code, so the data stays valid).
  * Element rotation U = Rx(gamma) @ Ry(beta) @ Rz(alpha) maps local -> global;
    the columnar stiff axis (local x3) is U[:,2] (stiff_axis). Assembly uses
    D_e = N^T D^H N with the strain Bond matrix N = voigt(U^T), equivalent to
    the 4th-order rotation C_glob = U.C_loc. Cross-validated in _rot_check.
  * NOTE: this deliberately deviates from the released TO_Spinodal code, whose
    Bond matrix (a) was missing the factor 2 in the three shear-row/normal-col
    entries (rows 4-6, col 1) -- breaking isotropy invariance at oblique
    angles -- and (b) effectively rotated by U^T, putting the stiff axis at
    U[2,:] instead of the U[:,2] used by its own orientation interpretation.
    Both are corrected here; see _rot_check.py.

All builders are vectorized over elements (leading axis n).
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import scipy.io

_PACKAGE_DIR = os.path.dirname(__file__)
_LEGACY_MAT = os.path.join(
    _PACKAGE_DIR, "..", "\u529b\u5b66\u9690\u8eab\u539f\u59cb\u4ee3\u7801",
    "materials", "coefficients_3d.mat"
)
_CANTILEVER_MAT = os.path.join(
    _PACKAGE_DIR, "..",
    "\u529b\u5b66\u9690\u8eabto\u60ac\u81c2\u6881\u67d4\u5ea6\u6700\u5c0f\u5316",
    "materials", "coefficients_3d.mat"
)
_MATERIAL_CANDIDATES = (
    os.path.join(_PACKAGE_DIR, "materials", "coefficients_3d.mat"),
    os.path.join(_PACKAGE_DIR, "..", "materials", "coefficients_3d.mat"),
    _LEGACY_MAT,
    _CANTILEVER_MAT,
)
_DEFAULT_MAT = next((p for p in _MATERIAL_CANDIDATES if os.path.exists(p)),
                    _MATERIAL_CANDIDATES[0])

_CLASS_KEYS = {
    "isotropic": "CH_isotropic",
    "lamellar": "CH_lamelar_x",
    "columnar": "CH_columnar_xy",
    "cubic": "CH_cubic",
}


def load_coeff(spinodal_class: str = "columnar", mat_path: str | None = None) -> np.ndarray:
    """Load the (6, 6, P) polynomial coefficient array for one spinodal class."""
    path = mat_path or _DEFAULT_MAT
    key = _CLASS_KEYS[spinodal_class]
    data = scipy.io.loadmat(path)
    if key not in data:
        raise KeyError(f"{key} not in {path}; available: "
                       f"{[k for k in data if not k.startswith('__')]}")
    coeff = np.asarray(data[key], dtype=float)
    if coeff.ndim != 3 or coeff.shape[:2] != (6, 6):
        raise ValueError(f"Unexpected coeff shape {coeff.shape}; expected (6,6,P).")
    return coeff


def ch_of_density(rho: np.ndarray, coeff: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Homogenized tensor and its rho-derivative for an array of densities.

    rho: (n,) ; coeff: (6,6,P) -> CH:(n,6,6), dCH/drho:(n,6,6).
    """
    rho = np.asarray(rho, dtype=float).ravel()
    P = coeff.shape[2]
    powers = np.arange(P - 1, -1, -1)                       # [P-1, ..., 1, 0]
    mp = rho[:, None] ** powers[None, :]                    # (n,P)
    CH = np.einsum("ijk,nk->nij", coeff, mp, optimize=True)

    # derivative: sum_{i=0..P-2} coeff[:,:,i] * (P-1-i) * rho**(P-2-i)
    dpow = powers[:-1]                                      # (P-1,)
    dmp = dpow[None, :] * rho[:, None] ** (dpow - 1)[None, :]
    dCH = np.einsum("ijk,nk->nij", coeff[:, :, :-1], dmp, optimize=True)
    return CH, dCH


def _voigt(U: np.ndarray) -> np.ndarray:
    """6x6 Bond/Voigt rotation matrix from 3x3 rotation U. U:(n,3,3)->(n,6,6).

    Vectorized via preallocation + direct element assignment (no np.stack)."""
    u00, u01, u02 = U[:, 0, 0], U[:, 0, 1], U[:, 0, 2]
    u10, u11, u12 = U[:, 1, 0], U[:, 1, 1], U[:, 1, 2]
    u20, u21, u22 = U[:, 2, 0], U[:, 2, 1], U[:, 2, 2]
    N = np.empty((U.shape[0], 6, 6))
    N[:, 0, 0] = u00*u00; N[:, 0, 1] = u01*u01; N[:, 0, 2] = u02*u02
    N[:, 0, 3] = u01*u02; N[:, 0, 4] = u02*u00; N[:, 0, 5] = u00*u01
    N[:, 1, 0] = u10*u10; N[:, 1, 1] = u11*u11; N[:, 1, 2] = u12*u12
    N[:, 1, 3] = u11*u12; N[:, 1, 4] = u12*u10; N[:, 1, 5] = u10*u11
    N[:, 2, 0] = u20*u20; N[:, 2, 1] = u21*u21; N[:, 2, 2] = u22*u22
    N[:, 2, 3] = u21*u22; N[:, 2, 4] = u22*u20; N[:, 2, 5] = u20*u21
    N[:, 3, 0] = 2*u10*u20; N[:, 3, 1] = 2*u11*u21; N[:, 3, 2] = 2*u12*u22
    N[:, 3, 3] = u12*u21+u11*u22; N[:, 3, 4] = u22*u10+u20*u12; N[:, 3, 5] = u20*u11+u21*u10
    N[:, 4, 0] = 2*u20*u00; N[:, 4, 1] = 2*u21*u01; N[:, 4, 2] = 2*u22*u02
    N[:, 4, 3] = u21*u02+u22*u01; N[:, 4, 4] = u20*u02+u22*u00; N[:, 4, 5] = u20*u01+u21*u00
    N[:, 5, 0] = 2*u00*u10; N[:, 5, 1] = 2*u01*u11; N[:, 5, 2] = 2*u02*u12
    N[:, 5, 3] = u01*u12+u02*u11; N[:, 5, 4] = u00*u12+u02*u10; N[:, 5, 5] = u00*u11+u01*u10
    return N


def _voigt_deriv(U: np.ndarray, dU: np.ndarray) -> np.ndarray:
    """d(voigt)/dt given U and dU/dt. U,dU:(n,3,3)->(n,6,6). Preallocated, no stack."""
    u00, u01, u02 = U[:, 0, 0], U[:, 0, 1], U[:, 0, 2]
    u10, u11, u12 = U[:, 1, 0], U[:, 1, 1], U[:, 1, 2]
    u20, u21, u22 = U[:, 2, 0], U[:, 2, 1], U[:, 2, 2]
    d00, d01, d02 = dU[:, 0, 0], dU[:, 0, 1], dU[:, 0, 2]
    d10, d11, d12 = dU[:, 1, 0], dU[:, 1, 1], dU[:, 1, 2]
    d20, d21, d22 = dU[:, 2, 0], dU[:, 2, 1], dU[:, 2, 2]
    R = np.empty((U.shape[0], 6, 6))
    R[:, 0, 0] = 2*u00*d00; R[:, 0, 1] = 2*u01*d01; R[:, 0, 2] = 2*u02*d02
    R[:, 0, 3] = u01*d02+u02*d01; R[:, 0, 4] = u02*d00+u00*d02; R[:, 0, 5] = u01*d00+u00*d01
    R[:, 1, 0] = 2*u10*d10; R[:, 1, 1] = 2*u11*d11; R[:, 1, 2] = 2*u12*d12
    R[:, 1, 3] = u11*d12+d11*u12; R[:, 1, 4] = u12*d10+d12*u10; R[:, 1, 5] = u10*d11+d10*u11
    R[:, 2, 0] = 2*u20*d20; R[:, 2, 1] = 2*u21*d21; R[:, 2, 2] = 2*u22*d22
    R[:, 2, 3] = u21*d22+d21*u22; R[:, 2, 4] = u22*d20+d22*u20; R[:, 2, 5] = u20*d21+d20*u21
    R[:, 3, 0] = 2*(u10*d20+d10*u20)
    R[:, 3, 1] = 2*u11*d21+2*d11*u21
    R[:, 3, 2] = 2*u12*d22+2*d12*u22
    R[:, 3, 3] = u12*d21+d12*u21+u11*d22+d11*u22
    R[:, 3, 4] = u22*d10+d22*u10+u20*d12+d20*u12
    R[:, 3, 5] = u20*d11+d20*u11+u21*d10+d21*u10
    R[:, 4, 0] = 2*(u20*d00+d20*u00)
    R[:, 4, 1] = 2*u21*d01+2*d21*u01
    R[:, 4, 2] = 2*u22*d02+2*d22*u02
    R[:, 4, 3] = u21*d02+d21*u02+u22*d01+d22*u01
    R[:, 4, 4] = u20*d02+d20*u02+u22*d00+d22*u00
    R[:, 4, 5] = u20*d01+d20*u01+u21*d00+d21*u00
    R[:, 5, 0] = 2*(u00*d10+d00*u10)
    R[:, 5, 1] = 2*u01*d11+2*d01*u11
    R[:, 5, 2] = 2*u02*d12+2*d02*u12
    R[:, 5, 3] = u01*d12+d01*u12+u02*d11+d02*u11
    R[:, 5, 4] = u00*d12+d00*u12+u02*d10+d02*u10
    R[:, 5, 5] = u00*d11+d00*u11+u01*d10+d01*u10
    return R


def _rot3(alpha, beta, gamma):
    """Per-element 3x3 rotation U=Rx(gamma)Ry(beta)Rz(alpha) and dU/d(angle).

    Preallocated rotation blocks (no nested np.stack)."""
    n = alpha.shape[0]
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    Rz = np.zeros((n, 3, 3)); Rz[:, 0, 0] = ca; Rz[:, 0, 1] = -sa
    Rz[:, 1, 0] = sa; Rz[:, 1, 1] = ca; Rz[:, 2, 2] = 1.0
    Ry = np.zeros((n, 3, 3)); Ry[:, 0, 0] = cb; Ry[:, 0, 2] = sb
    Ry[:, 1, 1] = 1.0; Ry[:, 2, 0] = -sb; Ry[:, 2, 2] = cb
    Rx = np.zeros((n, 3, 3)); Rx[:, 0, 0] = 1.0; Rx[:, 1, 1] = cg
    Rx[:, 1, 2] = -sg; Rx[:, 2, 1] = sg; Rx[:, 2, 2] = cg
    dRz = np.zeros((n, 3, 3)); dRz[:, 0, 0] = -sa; dRz[:, 0, 1] = -ca
    dRz[:, 1, 0] = ca; dRz[:, 1, 1] = -sa
    dRy = np.zeros((n, 3, 3)); dRy[:, 0, 0] = -sb; dRy[:, 0, 2] = cb
    dRy[:, 2, 0] = -cb; dRy[:, 2, 2] = -sb
    dRx = np.zeros((n, 3, 3)); dRx[:, 1, 1] = -sg; dRx[:, 1, 2] = -cg
    dRx[:, 2, 1] = cg; dRx[:, 2, 2] = -sg

    RyRz = Ry @ Rz
    U = Rx @ RyRz
    dU_da = Rx @ (Ry @ dRz)
    dU_db = Rx @ (dRy @ Rz)
    dU_dg = dRx @ RyRz
    return U, dU_da, dU_db, dU_dg


def stiff_axis(alpha, beta, gamma):
    """Global direction of the columnar stiff axis (local e3) per element: (n,3)."""
    alpha = np.asarray(alpha, float).ravel()
    beta = np.asarray(beta, float).ravel()
    gamma = np.asarray(gamma, float).ravel()
    U, *_ = _rot3(alpha, beta, gamma)
    return U[:, :, 2]   # U @ [0,0,1]


def principal_axis_angles(directions: np.ndarray):
    """Euler angles (alpha, beta, gamma) that align the columnar stiff axis
    (local e3) with each target direction -- the inverse of `stiff_axis`.

    With U = Rx(gamma) Ry(beta) Rz(alpha) the stiff axis is
        e3 = U @ [0,0,1] = [sin(beta), -sin(gamma) cos(beta), cos(gamma) cos(beta)]
    (independent of alpha: rotation about the column axis is mechanically
    near-degenerate for the transversely-isotropic columnar class). Inverting:
        beta  = arcsin(dx);  gamma = atan2(-dy, dz);  alpha = 0 (free).

    This depends only on _rot3 (the rotation construction), which the Voigt/Bond
    and node-ordering bug fixes never touched, so the inverse is consistent with
    the corrected forward map. Gated by `_orient_check.py`.

    directions: (n,3); need not be unit, sign irrelevant (a columnar axis is a
    line). Returns (alpha, beta, gamma), each (n,).
    """
    d = np.asarray(directions, float).reshape(-1, 3)
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.where(norm > 1e-12, norm, 1.0)
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    beta = np.arcsin(np.clip(dx, -1.0, 1.0))
    cosb = np.sqrt(np.maximum(1.0 - dx * dx, 0.0))
    safe = cosb > 1e-6                 # axis along x -> in-plane angle indeterminate
    gamma = np.where(safe, np.arctan2(-dy, dz), 0.0)
    alpha = np.zeros_like(beta)
    return alpha, beta, gamma


def principal_strain_directions(strain6: np.ndarray) -> np.ndarray:
    """Major-principal direction of each engineering-Voigt strain vector.

    strain6: (n,6) with order [exx, eyy, ezz, gyz, gxz, gxy] (engineering shear,
    matching the assembly B-matrix). Returns (n,3) unit eigenvectors for the
    largest-magnitude principal strain -- the "load path" direction used to seed
    columnar orientation.
    """
    e = np.asarray(strain6, float).reshape(-1, 6)
    n = e.shape[0]
    T = np.empty((n, 3, 3))
    T[:, 0, 0] = e[:, 0]; T[:, 1, 1] = e[:, 1]; T[:, 2, 2] = e[:, 2]
    T[:, 1, 2] = T[:, 2, 1] = 0.5 * e[:, 3]
    T[:, 0, 2] = T[:, 2, 0] = 0.5 * e[:, 4]
    T[:, 0, 1] = T[:, 1, 0] = 0.5 * e[:, 5]
    w, V = np.linalg.eigh(T)
    idx = np.argmax(np.abs(w), axis=1)
    return V[np.arange(n), :, idx]


def rotation_voigt(alpha, beta, gamma):
    """Return N and (dN/dalpha, dN/dbeta, dN/dgamma), each (n,6,6).

    N is built from U^T so that D_e = N^T D^H N equals the 4th-order tensor
    rotation C_glob = U.C_loc (vectors transform v_glob = U v_loc). This keeps
    the assembly consistent with stiff_axis() = U[:,2]: the columnar stiff
    local-x3 axis lands at U[:,2] in the global frame. Validated in _rot_check.
    """
    alpha = np.asarray(alpha, float).ravel()
    beta = np.asarray(beta, float).ravel()
    gamma = np.asarray(gamma, float).ravel()
    U, dU_da, dU_db, dU_dg = _rot3(alpha, beta, gamma)
    Ut = U.transpose(0, 2, 1)
    N = _voigt(Ut)
    dN_da = _voigt_deriv(Ut, dU_da.transpose(0, 2, 1))
    dN_db = _voigt_deriv(Ut, dU_db.transpose(0, 2, 1))
    dN_dg = _voigt_deriv(Ut, dU_dg.transpose(0, 2, 1))
    return N, dN_da, dN_db, dN_dg


if __name__ == "__main__":
    coeff = load_coeff("columnar")
    print(f"columnar coeff shape = {coeff.shape}  (polynomial order = {coeff.shape[2]-1})")
    # finite-difference check of dCH/drho
    rho = np.array([0.3, 0.5, 0.7])
    CH, dCH = ch_of_density(rho, coeff)
    eps = 1e-6
    CHp, _ = ch_of_density(rho + eps, coeff)
    CHm, _ = ch_of_density(rho - eps, coeff)
    err = np.max(np.abs((CHp - CHm) / (2 * eps) - dCH))
    print(f"dCH/drho FD max err = {err:.2e}")
    # finite-difference check of dN/dangle
    a = np.array([0.3]); b = np.array([-0.7]); g = np.array([1.1])
    N, dNa, dNb, dNg = rotation_voigt(a, b, g)
    Np, *_ = rotation_voigt(a + eps, b, g)
    Nm, *_ = rotation_voigt(a - eps, b, g)
    print(f"dN/dalpha FD max err = {np.max(np.abs((Np - Nm) / (2*eps) - dNa)):.2e}")
    print(f"N orthonormal-ish (N@iso check): see fea validation")
