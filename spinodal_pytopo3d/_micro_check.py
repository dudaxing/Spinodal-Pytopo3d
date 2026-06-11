"""Sanity checks for spinodal microstructure thresholding.

Run from project root:
  python -m spinodal_pytopo3d._micro_check

The paper's rho is solid volume fraction, so the generated solid set should use
phi <= sqrt(2)*erfinv(2*rho-1). These checks guard against accidentally flipping
the level-set sign in render/slice post-processing.
"""

from __future__ import annotations

import numpy as np
import scipy.special as special

from spinodal_pytopo3d.render_spinodal import (
    build_spinodal_field,
    global_phases,
    make_axis_coords,
    nearest_material_index_map,
    sample_columnar_wave_vectors_z,
)
from spinodal_pytopo3d.slice_print import slice_layer


def _cutoff(rho):
    return np.sqrt(2.0) * special.erfinv(2.0 * rho - 1.0)


def check_wave_vector_caps():
    """Columnar sampling must follow Eq. S4: >=85% of vectors inside the
    theta1=theta2=30deg cones around +-e1/+-e2 (15% uniform leakage, of which
    ~27% lands inside the cones by chance -> expected in-cone share ~89%)."""
    base = sample_columnar_wave_vectors_z(20000, 123)
    cos_t = np.cos(np.radians(30.0))
    in_cones = (np.abs(base[:, 0]) > cos_t) | (np.abs(base[:, 1]) > cos_t)
    frac = float(in_cones.mean())
    zmax = float(np.abs(base[in_cones, 2]).max())
    print(f"columnar wave vectors: {frac*100:.1f}% in Eq. S4 cones "
          f"(expect ~89%), max |v.e3| in cones = {zmax:.3f} (<= sin30 = 0.5)")
    return 0.85 <= frac <= 0.95 and zmax <= 0.5 + 1e-6


def check_normal_threshold():
    rng = np.random.default_rng(123)
    phi = rng.normal(size=1_000_000)
    errs = []
    for rho in (0.3, 0.5, 0.7):
        solid = float((phi <= _cutoff(rho)).mean())
        err = abs(solid - rho)
        errs.append(err)
        print(f"normal rho={rho:.1f}: solid={solid:.4f} err={err:.2e}")
    return max(errs) < 2e-3


def _one_cell_fields(frac_value):
    shape = (1, 1, 1)
    return {
        "Frac": np.full(shape, frac_value, dtype=np.float32),
        "alpha": np.zeros(shape, dtype=np.float32),
        "beta": np.zeros(shape, dtype=np.float32),
        "gamma": np.zeros(shape, dtype=np.float32),
    }


def check_build_spinodal_field():
    mask = np.ones((1, 1, 1), dtype=bool)
    spc = 24
    coords = [make_axis_coords(1, spc, 0.0) for _ in range(3)]
    phases = global_phases(20260608, 256)
    solids = []
    for rho in (0.3, 0.7):
        f = _one_cell_fields(rho)
        micro = build_spinodal_field(
            mask, f["Frac"], f["alpha"], f["beta"], f["gamma"],
            coords[0], coords[1], coords[2], m=3.0, n_waves=256,
            seed=20260608, phases=phases,
        )
        solid = float((micro >= 0).mean())
        solids.append(solid)
        print(f"field rho={rho:.1f}: solid={solid:.4f}")
    return solids[1] > solids[0] + 0.2


def check_slice_layer():
    mask = np.ones((1, 1, 1), dtype=bool)
    nearest = nearest_material_index_map(mask)
    n = 240
    xpx = (np.arange(n) + 0.5) / n
    ypx = (np.arange(n) + 0.5) / n
    sdf = -np.ones((2, 2, 2), dtype=np.float32)
    sdf_coords = (
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    )
    base = sample_columnar_wave_vectors_z(256, 20260608)
    phases = global_phases(20260608, 256)
    solids = []
    for rho in (0.3, 0.7):
        cured = slice_layer(
            0.5, xpx, ypx, _one_cell_fields(rho), nearest, base,
            np.float32(2 * np.pi * 3.0), 256, phases, sdf, sdf_coords,
        )
        solid = float(cured.mean())
        solids.append(solid)
        print(f"slice rho={rho:.1f}: solid={solid:.4f}")
    return solids[1] > solids[0] + 0.2


def main():
    checks = [
        ("wave vector cones (Eq. S4)", check_wave_vector_caps()),
        ("normal threshold", check_normal_threshold()),
        ("3D micro field", check_build_spinodal_field()),
        ("2D slice", check_slice_layer()),
    ]
    ok = all(v for _, v in checks)
    for name, passed in checks:
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
