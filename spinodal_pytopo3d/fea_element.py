"""
H8 hexahedral element machinery for spinodal (anisotropic, per-element) FEA,
made consistent with PyTopo3D's DOF/element ordering.

Why this module exists
----------------------
PyTopo3D assembles the global stiffness by scaling ONE isotropic element matrix
`lk_H8(nu)` by a scalar Young's modulus per element. The spinodal method instead
needs a DIFFERENT anisotropic 6x6 constitutive matrix `D_e` per element
(D_e = R(angles) . D^H_class(rho) . R^T), so each element has its own 24x24 `KE_e`.

To stay 100% compatible with PyTopo3D's `build_edof` (edofMat / iK / jK) and its
sparse-scatter assembly, the 24 local DOFs of our B-matrix must be ordered exactly
like PyTopo3D's local node order. We then *prove* consistency with a unit test:
KE assembled here with an isotropic D must equal `lk_H8(nu)` (see `validate()`).

Voigt convention (matches the original TO_Spinodal code and coefficients_3d.mat):
    strain = [exx, eyy, ezz, gyz, gxz, gxy]   (engineering shear).
"""

from __future__ import annotations

import numpy as np

# 2-point Gauss rule (weights = 1)
_GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)

# Local node natural coordinates (xi, eta, zeta) matching Liu & Tovar's lk_H8
# (the standard isoparametric H8 order: counter-clockwise bottom face z=-1, then
# top face z=+1). Verified against lk_H8 by `validate()` (exact match to 1e-12).
# Physical (x,y,z) corners with x<->xi, y<->eta, z<->zeta:
#   n0:(0,0,0) n1:(1,0,0) n2:(1,1,0) n3:(0,1,0)  (bottom)
#   n4:(0,0,1) n5:(1,0,1) n6:(1,1,1) n7:(0,1,1)  (top)
_NAT = np.array(
    [
        [-1.0, -1.0, -1.0],  # n0
        [ 1.0, -1.0, -1.0],  # n1
        [ 1.0,  1.0, -1.0],  # n2
        [-1.0,  1.0, -1.0],  # n3
        [-1.0, -1.0,  1.0],  # n4
        [ 1.0, -1.0,  1.0],  # n5
        [ 1.0,  1.0,  1.0],  # n6
        [-1.0,  1.0,  1.0],  # n7
    ]
)


def _shape_grad_natural(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Trilinear H8 shape-function gradients d N_a / d(xi,eta,zeta), shape (8, 3)."""
    xa, ya, za = _NAT[:, 0], _NAT[:, 1], _NAT[:, 2]
    g = np.empty((8, 3))
    g[:, 0] = 0.125 * xa * (1.0 + eta * ya) * (1.0 + zeta * za)
    g[:, 1] = 0.125 * ya * (1.0 + xi * xa) * (1.0 + zeta * za)
    g[:, 2] = 0.125 * za * (1.0 + xi * xa) * (1.0 + eta * ya)
    return g


def element_B_integrated(h: float = 1.0) -> np.ndarray:
    """
    Return the integration-weighted B matrices for a regular cube element of size h.

    Returns
    -------
    np.ndarray, shape (8, 6, 24)
        B_int[g] = sqrt(w_g * detJ) * B(gp_g), so that
        KE = sum_g B_int[g].T @ D @ B_int[g].
    """
    jac = h / 2.0                 # x = (xi + 1) * h / 2  ->  dx/dxi = h/2
    detj = jac ** 3
    inv_jac = 1.0 / jac           # dN/dx = dN/dxi * (1/jac)

    bmats = []
    for zeta in _GP:
        for eta in _GP:
            for xi in _GP:
                g = _shape_grad_natural(xi, eta, zeta) * inv_jac  # (8,3) physical grads
                b = np.zeros((6, 24))
                b[0, 0::3] = g[:, 0]                       # exx
                b[1, 1::3] = g[:, 1]                       # eyy
                b[2, 2::3] = g[:, 2]                       # ezz
                b[3, 1::3] = g[:, 2]; b[3, 2::3] = g[:, 1]  # gyz
                b[4, 0::3] = g[:, 2]; b[4, 2::3] = g[:, 0]  # gxz
                b[5, 0::3] = g[:, 1]; b[5, 1::3] = g[:, 0]  # gxy
                w = 1.0
                bmats.append(np.sqrt(w * detj) * b)
    return np.stack(bmats, axis=0)


def isotropic_D(E: float = 1.0, nu: float = 0.3) -> np.ndarray:
    """Isotropic 6x6 constitutive matrix in [xx,yy,zz,yz,xz,xy] (engineering shear)."""
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    s = (1.0 - 2.0 * nu) / 2.0
    return c * np.array(
        [
            [1 - nu, nu, nu, 0, 0, 0],
            [nu, 1 - nu, nu, 0, 0, 0],
            [nu, nu, 1 - nu, 0, 0, 0],
            [0, 0, 0, s, 0, 0],
            [0, 0, 0, 0, s, 0],
            [0, 0, 0, 0, 0, s],
        ]
    )


def element_KE(D: np.ndarray, B_int: np.ndarray) -> np.ndarray:
    """KE = sum_g B_g^T D B_g  for a single element. D:(6,6), B_int:(8,6,24)."""
    return np.einsum("gsd,st,gte->de", B_int, D, B_int, optimize=True)


def element_KE_batch(D_all: np.ndarray, B_int: np.ndarray) -> np.ndarray:
    """Per-element KE for a stack of constitutive matrices.

    D_all:(nele,6,6), B_int:(8,6,24) -> (nele,24,24)."""
    return np.einsum("gsd,est,gtf->edf", B_int, D_all, B_int, optimize=True)


def validate(nu: float = 0.3, tol: float = 1e-9) -> float:
    """
    Correctness gate: our hand-built isotropic KE must equal PyTopo3D's lk_H8(nu).

    Returns the max abs difference and raises AssertionError if it exceeds `tol`.
    Requires PyTopo3D-main to be importable (the caller adds it to sys.path).
    """
    from pytopo3d.utils.stiffness import lk_H8

    B_int = element_B_integrated(1.0)
    KE_ours = element_KE(isotropic_D(1.0, nu), B_int)
    KE_ref = lk_H8(nu)
    diff = float(np.max(np.abs(KE_ours - KE_ref)))
    assert diff < tol, (
        f"KE mismatch vs lk_H8: max|diff|={diff:.3e} (tol={tol:.1e}). "
        "Node ordering / Voigt convention is inconsistent with PyTopo3D."
    )
    return diff


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "PyTopo3D-main"))
    d = validate()
    print(f"[OK] KE matches lk_H8, max|diff| = {d:.3e}")
