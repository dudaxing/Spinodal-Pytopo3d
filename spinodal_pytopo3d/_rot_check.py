"""
Cross-validation of the 6x6 Voigt/Bond rotation used in assembly.

The element constitutive matrix is D_e = N^T D^H N with N = voigt(U),
U = Rx(gamma) Ry(beta) Rz(alpha). This check rebuilds the same rotation the
slow-but-unambiguous way -- convert D^H (engineering-shear Voigt) to the full
3x3x3x3 tensor, rotate with np.einsum, convert back -- and compares:

  1. einsum equivalence:   N^T D N  ==  voigt4(R . C . R^T x4)  at random angles;
  2. isotropy invariance:  rotating the rho=1 isotropic solid must not change it;
  3. 90-degree axis swap:  beta=90deg about x2 maps the columnar stiff local x3
     axis onto global x1, so D_e[0,0] == D^H[2,2] (consistent with stiff_axis).

Run:  python -m spinodal_pytopo3d._rot_check
"""

from __future__ import annotations

import sys

import numpy as np

from spinodal_pytopo3d import spinodal_material as sm

# Voigt order [xx, yy, zz, yz, xz, xy] (engineering shear). For a stiffness
# matrix in this convention, D[a, b] = C[i(a), j(a), k(b), l(b)] exactly.
_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def voigt_to_tensor(D):
    C = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(_PAIRS):
        for b, (k, l) in enumerate(_PAIRS):
            C[i, j, k, l] = C[j, i, k, l] = C[i, j, l, k] = C[j, i, l, k] = D[a, b]
    return C


def tensor_to_voigt(C):
    D = np.empty((6, 6))
    for a, (i, j) in enumerate(_PAIRS):
        for b, (k, l) in enumerate(_PAIRS):
            D[a, b] = C[i, j, k, l]
    return D


def rotate_tensor(C, R):
    return np.einsum("im,jn,ko,lp,mnop->ijkl", R, R, R, R, C)


def reference_rotation(D, U):
    """Rotate a local-frame stiffness to the global frame via the 4th-order
    tensor route: C_glob = U.C_loc (vectors transform as v_glob = U v_loc)."""
    return tensor_to_voigt(rotate_tensor(voigt_to_tensor(D), U))


def main():
    rng = np.random.default_rng(7)
    coeff = sm.load_coeff("columnar")
    D_aniso = sm.ch_of_density(np.array([0.5]), coeff)[0][0]   # (6,6), anisotropic
    D_iso = sm.ch_of_density(np.array([1.0]), coeff)[0][0]     # isotropic solid
    ok = True

    # --- 1. einsum equivalence at random oblique angles ---------------------
    worst = 0.0
    for _ in range(20):
        a, b, g = rng.uniform(-np.pi, np.pi, 3)
        N = sm.rotation_voigt([a], [b], [g])[0][0]
        U = sm._rot3(np.array([a]), np.array([b]), np.array([g]))[0][0]
        D_fast = N.T @ D_aniso @ N
        D_ref = reference_rotation(D_aniso, U)
        worst = max(worst, np.abs(D_fast - D_ref).max())
    print(f"einsum equivalence: max |N^T D N - einsum| = {worst:.3e}")
    if worst > 1e-10:
        ok = False
        # localize the discrepancy for the last sample
        bad = np.argwhere(np.abs(D_fast - D_ref) > 1e-10)
        print(f"  FAIL: mismatched Voigt entries: {sorted(set(map(tuple, bad)))}")

    # --- 2. isotropy invariance ---------------------------------------------
    # Exact isotropic stiffness (E=1, nu=0.3, engineering shear): a proper
    # rotation must leave it untouched. The fitted rho=1 data deviates from
    # perfect isotropy by a small amount -- reported as info, not a failure.
    E, nu = 1.0, 0.3
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D_exact = np.diag([lam + 2 * mu] * 3 + [mu] * 3)
    D_exact[:3, :3] += lam * (1 - np.eye(3))
    worst = 0.0
    for _ in range(20):
        a, b, g = rng.uniform(-np.pi, np.pi, 3)
        N = sm.rotation_voigt([a], [b], [g])[0][0]
        worst = max(worst, np.abs(N.T @ D_exact @ N - D_exact).max())
    print(f"isotropy invariance: max |N^T D_iso N - D_iso| = {worst:.3e} "
          f"(fitted rho=1 data intrinsic anisotropy: "
          f"{np.abs(D_iso - D_exact).max():.1e})")
    if worst > 1e-10:
        ok = False
        print("  FAIL: rotating the isotropic solid changed it")

    # --- 3. 90-degree stiff-axis swap (consistency with stiff_axis) ---------
    a, b, g = 0.0, np.pi / 2.0, 0.0          # local x3 -> global x1
    axis = sm.stiff_axis([a], [b], [g])[0]
    N = sm.rotation_voigt([a], [b], [g])[0][0]
    D_rot = N.T @ D_aniso @ N
    err_axis = np.abs(axis - [1, 0, 0]).max()
    err_swap = max(abs(D_rot[0, 0] - D_aniso[2, 2]), abs(D_rot[2, 2] - D_aniso[0, 0]))
    print(f"90deg swap: stiff_axis err = {err_axis:.3e}, "
          f"D_e[0,0]<->D[2,2] err = {err_swap:.3e}")
    if err_axis > 1e-12 or err_swap > 1e-10:
        ok = False
        print("  FAIL: assembly rotation inconsistent with stiff_axis convention")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
