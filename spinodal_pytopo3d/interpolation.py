"""
Macro-scale material interpolation for spinodal topology optimization.

In the single-class spinodal formulation, each element carries:
  * a macro "presence" variable z  (SIMP + Heaviside projection -> solid/void),
  * a spinodal density rho in [rho_min, rho_max]  (handled in spinodal_material.py),
  * three orientation angles.

The element constitutive matrix is  D_e = E_simp(z_tilde) * R . D^H(rho) . R^T,
where E_simp is the scalar computed here. This mirrors the SIMP-Heaviside path of
the original `Material_Interpolation_Func` in TO_Spinodal (single material, so the
DMO/gamma mixing term degenerates to 1 and is dropped).

Physical (projected) density rho_bar = Heaviside(rho_tilde) is also the quantity
used in the global volume measure  V = rho_bar * rho_spinodal.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def heaviside_projection(
    rho_tilde: np.ndarray, beta: float, eta: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth Heaviside projection (Wang/Guest form) and its derivative.

    rho_bar = (tanh(beta*eta) + tanh(beta*(rho_tilde - eta)))
              / (tanh(beta*eta) + tanh(beta*(1 - eta)))

    Returns (rho_bar, d rho_bar / d rho_tilde).
    """
    denom = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    rho_bar = (np.tanh(beta * eta) + np.tanh(beta * (rho_tilde - eta))) / denom
    d_rho_bar = beta * (1.0 - np.tanh(beta * (rho_tilde - eta)) ** 2) / denom
    return rho_bar, d_rho_bar


def simp_stiffness(
    rho_bar: np.ndarray, penal: float, Emin: float = 1e-9
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SIMP modulus scalar E in [Emin, 1] and its derivative w.r.t. rho_bar.

    E = Emin + (1 - Emin) * rho_bar**penal
    """
    E = Emin + (1.0 - Emin) * rho_bar ** penal
    dE = (1.0 - Emin) * penal * np.clip(rho_bar, 1e-12, None) ** (penal - 1.0)
    return E, dE


def macro_interpolation(
    rho_tilde: np.ndarray, penal: float, beta: float, eta: float = 0.5, Emin: float = 1e-9
):
    """
    Full macro chain: rho_tilde -> (E_simp, dE/drho_tilde, rho_bar, drho_bar/drho_tilde).

    E_simp scales the (rotated) homogenized spinodal tensor; rho_bar feeds the
    volume constraint. Derivatives are w.r.t. the filtered field rho_tilde; the
    chain rule back to the raw variable z is closed by the filter transpose.
    """
    rho_bar, d_rho_bar = heaviside_projection(rho_tilde, beta, eta)
    E, dE_drho_bar = simp_stiffness(rho_bar, penal, Emin)
    dE_drho_tilde = dE_drho_bar * d_rho_bar
    return E, dE_drho_tilde, rho_bar, d_rho_bar
