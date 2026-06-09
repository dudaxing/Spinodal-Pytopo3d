"""
Optimizer for single-class spinodal compliance minimization.

The design vector is the concatenation [z, Frac, alpha, beta, gamma] (each nele).
The update is the augmented-Lagrangian + normalized-gradient scheme used by the
original TO_Spinodal code (one global volume constraint), ported to numpy.

Continuation follows the Senhora 2022 / SI schedule:
  * SIMP penalty p = 3 (fixed);
  * Heaviside sharpness beta starts at 1.0 and, after `beta_start_iter` iterations,
    is multiplied by `beta_growth` every `beta_period` iterations up to `beta_max`.
    (This is the schedule the paper describes and is applied here correctly, unlike
    the cantilever-port regression noted in 项目分析与工作日志.md §3.5.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .spinodal_fea import SpinodalFEA, SpinodalParams


class ALGradientUpdater:
    """Augmented-Lagrangian + normalized-gradient design update (ported)."""

    def __init__(self, lower, upper):
        self.lower = lower.reshape(-1, 1)
        self.upper = upper.reshape(-1, 1)
        self.relaxation = 1.0
        self.f_old = 1e7
        self.f_old_2 = 1e7
        self._init = False

    def update(self, z0, f, df_dz, g, dg_dz, it, move_vector):
        z0 = z0.reshape(-1, 1)
        df_dz = df_dz.reshape(z0.shape).copy()
        g = np.reshape(np.asarray(g, dtype=float), (1, -1))
        n_const = g.size
        dg_dz = dg_dz.reshape(z0.shape[0], n_const)

        if not self._init:
            self.lamda = np.zeros((1, n_const))
            self.mu = 1.0
            self.g_accum = 0.0
            self.mu_update_iter = 0
            self.normalize_internal = 1
            self._init = True

        if self.normalize_internal == 1:
            self.df_0 = max(np.mean(np.abs(df_dz)), 1e-12)
            self.mu = min(self.mu * 1.005, 1e5)
        df_dz /= self.df_0

        if n_const > 0:
            if it % 5 == 0:
                self.lamda = np.maximum(self.lamda + self.mu * g, 0)
                if (g > 0.01).any():
                    self.mu = min(self.mu * 1.1, 1e6)
            if (g > 0.01).any():
                self.mu_update_iter += 1
                self.g_accum += np.amax(g)
                if self.g_accum > 1 and self.mu_update_iter > 20:
                    self.mu = min(self.mu * 10.0, 1e6)
                    self.g_accum = 0.0
                    self.mu_update_iter = 0
            if (g < 0).all():
                self.g_accum = 0.0
            penal = np.sum(
                (self.lamda + self.mu * np.maximum(g, -self.lamda / self.mu)) * dg_dz,
                axis=1,
            )
            dL = df_dz + np.expand_dims(penal, axis=1)
        else:
            dL = df_dz

        if self.normalize_internal == 1:
            self.dL_0 = max(np.mean(np.abs(dL)), 1e-12)
            self.normalize_internal = 0

        if n_const > 0:
            if (self.f_old - self.f_old_2) * (f - self.f_old) < 0:
                self.relaxation = max(self.relaxation * 0.25, 1e-5)
            else:
                self.relaxation = min(self.relaxation * 1.25, 1.0)
        else:
            if f > self.f_old or (self.f_old - self.f_old_2) * (f - self.f_old) < 0:
                self.relaxation = max(self.relaxation * 0.5, 1e-5)
            else:
                self.relaxation = min(self.relaxation * 1.1, 1.0)

        mv = move_vector.reshape(z0.shape)
        lo = np.maximum(self.lower, z0 - mv)
        hi = np.minimum(self.upper, z0 + mv)
        dz = dL / self.dL_0
        z_new = np.maximum(np.minimum(z0 - dz * self.relaxation, hi), lo)

        self.f_old_2 = self.f_old
        self.f_old = f
        return z_new.ravel()


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class OptOptions:
    volfrac: float = 0.05
    penal: float = 3.0
    rho_min: float = 0.3
    rho_max: float = 0.7
    optimize_density: bool = True       # False -> fixed spinodal density (shell case)
    move_z: float = 0.1
    move_frac: float = 0.05
    move_angle: float = 0.2             # radians
    max_iter: int = 300
    tol: float = 5e-4
    eta: float = 0.5
    Emin: float = 1e-9
    beta0: float = 1.0
    beta_start_iter: int = 150
    beta_period: int = 30
    beta_growth: float = 1.25
    beta_max: float = 15.0
    beta_add: float = 0.0               # >0: additive beta step (paper); else multiplicative
    penal_steps: tuple = ()             # SIMP p continuation, e.g. (1,1.5,2,2.5,3); ()=fixed
    penal_iters: tuple = ()             # iters per continuation step, e.g. (150,100,100,50,50)
    # dedicated orientation optimization (aligns columnar stiff axis to stress)
    angle_subiters: int = 15
    angle_period: int = 20
    angle_phase_frac: float = 0.7       # run angle sub-iters for it < frac*max_iter
    angle_phase_iter: Optional[int] = None
    verbose: bool = True
    log_every: int = 10


def _angle_subiterations(fea, state, params, angle_updater, move_vec, obj0, opt):
    """Run orientation-only gradient steps (z and Frac frozen)."""
    z, Frac, alpha, beta, gamma = state
    nele = fea.nele
    g0 = np.zeros((1, 0))
    dg0 = np.zeros((3 * nele, 0))
    # z and Frac are frozen here -> precompute the macro/material fields once
    cache = fea.macro_material_cache(z, Frac, params)
    for s in range(opt.angle_subiters):
        o = fea.analyze(z, Frac, alpha, beta, gamma, params, cache=cache)
        x_old = np.concatenate([alpha, beta, gamma])
        df = np.concatenate([o["dc_da"], o["dc_db"], o["dc_dg"]]) / obj0
        x_new = angle_updater.update(x_old, o["c"] / obj0, df, g0, dg0, s, move_vec)
        alpha = _wrap(x_new[:nele])
        beta = _wrap(x_new[nele:2 * nele])
        gamma = _wrap(x_new[2 * nele:])
    return alpha, beta, gamma


def optimize(fea: SpinodalFEA, opt: OptOptions):
    nele = fea.nele
    rho0 = 0.5 * (opt.rho_min + opt.rho_max)

    # --- initial design (satisfy volume: presence*rho ~ volfrac) ---
    frac0 = opt.rho_min if not opt.optimize_density else rho0
    z = np.full(nele, min(opt.volfrac / max(frac0, 1e-9), 1.0))
    Frac = np.full(nele, frac0)
    alpha = np.zeros(nele); beta = np.zeros(nele); gamma = np.zeros(nele)

    # --- bounds & move vector (order: z, Frac, a, b, g) ---
    lower = np.concatenate([np.zeros(nele), np.full(nele, opt.rho_min),
                            np.full(3 * nele, -np.pi)])
    upper = np.concatenate([np.ones(nele), np.full(nele, opt.rho_max),
                            np.full(3 * nele, np.pi)])
    mv_frac = opt.move_frac if opt.optimize_density else 0.0
    move_vector = np.concatenate([np.full(nele, opt.move_z), np.full(nele, mv_frac),
                                  np.full(3 * nele, opt.move_angle)])

    updater = ALGradientUpdater(lower, upper)
    angle_updater = ALGradientUpdater(
        np.full(3 * nele, -np.pi), np.full(3 * nele, np.pi)
    )
    angle_mv = np.full(3 * nele, opt.move_angle)
    angle_phase_end = (int(opt.angle_phase_iter) if opt.angle_phase_iter is not None
                       else int(opt.angle_phase_frac * opt.max_iter))
    p0 = opt.penal_steps[0] if opt.penal_steps else opt.penal
    params = SpinodalParams(penal=p0, beta=opt.beta0, eta=opt.eta, Emin=opt.Emin)

    # SIMP p-continuation (paper: p=[1,1.5,2,2.5,3] over [150,100,100,50,50] iters);
    # Heaviside beta ramps only after continuation completes.
    cont_thresh = np.cumsum(opt.penal_iters) if opt.penal_steps else np.array([])
    beta_start = int(cont_thresh[-1]) if opt.penal_steps else opt.beta_start_iter

    hist = {"c": [], "vol": [], "g": [], "change": [], "beta": [], "penal": []}
    obj0 = None
    last = None

    for it in range(opt.max_iter):
        if opt.penal_steps:
            k = int(np.searchsorted(cont_thresh, it, side="right"))
            params.penal = opt.penal_steps[min(k, len(opt.penal_steps) - 1)]
        # Heaviside continuation (additive per paper if beta_add>0, else multiplicative)
        if it > beta_start and (it - beta_start) % opt.beta_period == 0:
            if opt.beta_add > 0:
                params.beta = min(params.beta + opt.beta_add, opt.beta_max)
            else:
                params.beta = min(params.beta * opt.beta_growth, opt.beta_max)

        # dedicated orientation optimization (z, Frac frozen)
        if (opt.angle_subiters > 0 and it > 0 and it % opt.angle_period == 0
                and it < angle_phase_end and obj0 is not None):
            alpha, beta, gamma = _angle_subiterations(
                fea, (z, Frac, alpha, beta, gamma), params, angle_updater,
                angle_mv, obj0, opt
            )

        out = fea.analyze(z, Frac, alpha, beta, gamma, params)
        c, vol = out["c"], out["vol"]
        if obj0 is None:
            obj0 = max(abs(c), 1e-12)

        g = vol - opt.volfrac
        df = np.concatenate([out["dc_dz"], out["dc_dFrac"],
                             out["dc_da"], out["dc_db"], out["dc_dg"]]) / obj0
        dg = np.concatenate([out["dvol_dz"], out["dvol_dFrac"], np.zeros(3 * nele)])

        x_old = np.concatenate([z, Frac, alpha, beta, gamma])
        x_new = updater.update(x_old, c / obj0, df, g, dg, it, move_vector)

        z = x_new[:nele]
        Frac = x_new[nele:2 * nele]
        alpha = _wrap(x_new[2 * nele:3 * nele])
        beta = _wrap(x_new[3 * nele:4 * nele])
        gamma = _wrap(x_new[4 * nele:5 * nele])

        change = float(np.mean(np.abs(x_new - x_old)))
        hist["c"].append(c); hist["vol"].append(vol); hist["g"].append(g)
        hist["change"].append(change); hist["beta"].append(params.beta)
        hist["penal"].append(params.penal)
        last = out

        if opt.verbose and (it % opt.log_every == 0 or it == opt.max_iter - 1):
            print(f"it {it:4d} | c={c:.4e} | c/c0={c/obj0:.4f} | vol={vol:.4f} "
                  f"| g={g:+.2e} | p={params.penal:.1f} | beta={params.beta:.2f} | chg={change:.2e}")

        # converge only once the Heaviside projection is (nearly) fully sharpened
        # and the volume constraint is satisfied -- avoids stopping the instant the
        # volume is first met, before orientation/topology settle.
        if (change < opt.tol and params.beta >= opt.beta_max - 1e-9
                and g < 1e-3):
            if opt.verbose:
                print(f"Converged at it {it} (change {change:.2e}, beta={params.beta:.1f}).")
            break

    return dict(z=z, Frac=Frac, alpha=alpha, beta=beta, gamma=gamma,
                c=last["c"], vol=last["vol"], rho_bar=last["rho_bar"],
                E_simp=last["E_simp"], V=last["V"], history=hist, params=params)
