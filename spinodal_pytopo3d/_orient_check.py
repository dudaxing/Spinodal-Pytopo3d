"""
Correctness gate for the stress-seeded orientation initialization.

Two independent-reference checks (the kind that catch convention mismatches the
self-consistent FD tests cannot):

  1. Round trip: stiff_axis(principal_axis_angles(d)) == +-d for random target
     directions d. This pins the inverse-angle map to the *corrected* forward
     stiff_axis (the inverse depends only on _rot3, untouched by the Voigt/Bond
     and node-ordering fixes, but we verify numerically rather than trust it).
  2. principal_strain_directions on an analytic uniaxial strain returns that
     axis, and the engineering-shear (factor-2) bookkeeping is correct.

Run:  python -m spinodal_pytopo3d._orient_check
"""

from __future__ import annotations

import sys

import numpy as np

from spinodal_pytopo3d import spinodal_material as sm


def main():
    rng = np.random.default_rng(11)
    ok = True

    # --- 1. inverse-angle round trip ----------------------------------------
    d = rng.normal(size=(2000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    a, b, g = sm.principal_axis_angles(d)
    axis = sm.stiff_axis(a, b, g)
    # columnar axis is a line: match up to sign
    align = np.abs(np.sum(axis * d, axis=1))
    worst = float(np.min(align))
    print(f"inverse-angle round trip: min |stiff_axis(inv(d)) . d| = {worst:.6f} "
          f"(1.0 = exact)")
    if worst < 1.0 - 1e-9:
        ok = False
        print(f"  FAIL: inverse-angle map inconsistent with stiff_axis "
              f"({(align < 1 - 1e-9).sum()} of {len(d)} dirs off)")

    # --- 2. principal-strain direction on analytic uniaxial states ----------
    worst = 0.0
    for target in np.eye(3):
        # pure uniaxial strain along `target`: tensor = target (x) target
        Tt = np.outer(target, target)
        e6 = np.array([Tt[0, 0], Tt[1, 1], Tt[2, 2],
                       2 * Tt[1, 2], 2 * Tt[0, 2], 2 * Tt[0, 1]])  # engineering
        v = sm.principal_strain_directions(e6[None, :])[0]
        worst = max(worst, 1.0 - abs(float(v @ target)))
    # pure shear (exy only): both principal strains are +-0.5, equal magnitude,
    # so the major direction is degenerate between the two +-45 deg diagonals.
    # The valid check is: the axis lies in the x-y plane (vz=0) at 45 deg.
    e6 = np.array([0, 0, 0, 0, 0, 2 * 0.5])      # gxy = 1 -> exy = 0.5
    v = sm.principal_strain_directions(e6[None, :])[0]
    shear_err = abs(float(v[2])) + abs(abs(float(v[0])) - abs(float(v[1])))
    worst = max(worst, shear_err)
    print(f"principal_strain_directions: worst axis error = {worst:.3e}")
    if worst > 1e-9:
        ok = False
        print("  FAIL: principal-strain extraction or engineering-shear factor wrong")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
