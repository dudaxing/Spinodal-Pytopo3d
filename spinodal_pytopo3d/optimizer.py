"""
Optimizer for single-class spinodal compliance minimization.

The design vector is the concatenation [z, Frac, alpha, beta, gamma] (each nele).

Two augmented-Lagrangian update schemes are available:
  * ALGradientUpdater (default): the normalized-gradient scheme used by the
    *released* TO_Spinodal code (conditional mu growth, oscillation-adaptive
    relaxation), ported to numpy.
  * SIGradientUpdater (`OptOptions.si_schedule=True`): the scheme the paper's
    Supporting Information actually specifies (Eq. S10-S12): unconditional
    mu <- 1.25*mu and lambda <- lambda + mu*max(g, -lambda/mu) every 5 inner
    iterations, and deterministic step decay tau(k+1) = max(0.99*tau(k), 0.01).
    Combine with `cont_tol=0.02` (per-step early advance on max|dz|, SI S4.1)
    and `beta0=0.1` (SI initial Heaviside sharpness) for the full SI schedule.

Continuation (both schemes): SIMP p-steps via `penal_steps`/`penal_iters`
(paper: p=[1,1.5,2,2.5,3] over [150,100,100,50,50]); Heaviside beta ramps only
after p-continuation completes (additive +0.5 every 15 iterations up to 25 in
the paper's setting).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import spinodal_material as sm
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


class SIGradientUpdater:
    """AL + gradient-descent update exactly per the paper's SI (Eq. S10-S12).

    Differences from the legacy (released TO_Spinodal code) updater above:
      * Eq. S11: every 5 inner iterations, lambda <- lambda + mu*max(g, -lambda/mu)
        and mu <- 1.25*mu UNCONDITIONALLY (legacy grows mu only while g > 0.01,
        and far slower -- the main reason its volume constraint converges slowly);
      * Eq. S12: deterministic step decay tau(0)=1, tau(k+1)=max(0.99*tau(k), 0.01)
        (legacy uses an oscillation-detection relaxation heuristic instead).

    The SI does not specify gradient normalization; as in the legacy updater the
    objective/Lagrangian gradients are scaled by their first-iteration mean
    magnitude so tau=1 combined with the move-limit clamp is meaningful across
    problem scales.
    """

    def __init__(self, lower, upper):
        self.lower = lower.reshape(-1, 1)
        self.upper = upper.reshape(-1, 1)
        self.tau = 1.0
        self._init = False

    def update(self, z0, f, df_dz, g, dg_dz, it, move_vector):
        z0 = z0.reshape(-1, 1)
        df = df_dz.reshape(z0.shape).copy()
        g = np.reshape(np.asarray(g, dtype=float), (1, -1))
        n_const = g.size
        dg = dg_dz.reshape(z0.shape[0], n_const)

        if not self._init:
            self.lamda = np.zeros((1, n_const))
            self.mu = 1.0
            self.df_0 = max(np.mean(np.abs(df)), 1e-12)
            self.dL_0 = None
            self._init = True
        df /= self.df_0

        if n_const > 0:
            if it > 0 and it % 5 == 0:                      # Eq. S11
                self.lamda = self.lamda + self.mu * np.maximum(g, -self.lamda / self.mu)
                # The SI puts no cap on mu, but unbounded 1.25^k growth (1e23
                # over ~1200 iterations) makes the penalty gradient flip-flop
                # the design each iteration once tau hits its floor (observed
                # limit cycle). Cap at the legacy updater's 1e6; lambda keeps
                # adapting, which is what drives AL convergence.
                self.mu = min(self.mu * 1.25, 1e6)
            w = self.lamda + self.mu * np.maximum(g, -self.lamda / self.mu)
            dL = df + np.expand_dims(np.sum(w * dg, axis=1), axis=1)
        else:
            dL = df

        if self.dL_0 is None:
            self.dL_0 = max(np.mean(np.abs(dL)), 1e-12)

        mv = move_vector.reshape(z0.shape)                  # Eq. S12
        lo = np.maximum(self.lower, z0 - mv)
        hi = np.minimum(self.upper, z0 + mv)
        z_new = np.maximum(np.minimum(z0 - (dL / self.dL_0) * self.tau, hi), lo)
        self.tau = max(0.99 * self.tau, 0.01)
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
    si_schedule: bool = False           # SI-exact AL schedule (Eq. S11/S12 updater)
    cont_tol: float = 0.0               # >0: advance continuation early when max|dz| < tol
                                        # (SI S4.1: tol=0.02 on the selection variables)
    init_orient_from_stress: bool = False  # seed angles from the principal-strain
                                        # direction of one initial solve instead of 0
                                        # (helps the non-convex orientation problem;
                                        # opt-in, keeps default runs bit-identical)
    # dedicated orientation optimization (aligns columnar stiff axis to stress)
    angle_subiters: int = 15
    angle_period: int = 20
    angle_phase_frac: float = 0.7       # run angle sub-iters for it < frac*max_iter
    angle_phase_iter: Optional[int] = None
    passive_z: Optional[np.ndarray] = None
    passive_frac_value: Optional[float] = None
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

    # --- optional: seed orientation from the initial load-path direction ---
    if opt.init_orient_from_stress:
        p0_seed = opt.penal_steps[0] if opt.penal_steps else opt.penal
        p_seed = SpinodalParams(penal=p0_seed, beta=opt.beta0, eta=opt.eta,
                                Emin=opt.Emin)
        seed = fea.analyze(z, Frac, alpha, beta, gamma, p_seed, return_strain=True)
        dirs = sm.principal_strain_directions(seed["eps_avg"])
        alpha, beta, gamma = sm.principal_axis_angles(dirs)
        alpha = _wrap(alpha); beta = _wrap(beta); gamma = _wrap(gamma)
        if opt.verbose:
            print("[init] orientation seeded from principal-strain directions")

    passive_z = np.zeros(nele, dtype=bool)
    if opt.passive_z is not None:
        passive_z = np.asarray(opt.passive_z, dtype=bool).ravel()
        if passive_z.size != nele:
            raise ValueError(f"passive_z has {passive_z.size} entries, expected {nele}")
        z[passive_z] = 1.0
        if opt.passive_frac_value is not None:
            Frac[passive_z] = float(np.clip(opt.passive_frac_value, opt.rho_min, opt.rho_max))

    # --- bounds & move vector (order: z, Frac, a, b, g) ---
    z_lower = np.zeros(nele); z_upper = np.ones(nele)
    frac_lower = np.full(nele, opt.rho_min); frac_upper = np.full(nele, opt.rho_max)
    if passive_z.any():
        z_lower[passive_z] = 1.0
        z_upper[passive_z] = 1.0
        if opt.passive_frac_value is not None:
            frac_value = float(np.clip(opt.passive_frac_value, opt.rho_min, opt.rho_max))
            frac_lower[passive_z] = frac_value
            frac_upper[passive_z] = frac_value
    lower = np.concatenate([z_lower, frac_lower, np.full(3 * nele, -np.pi)])
    upper = np.concatenate([z_upper, frac_upper, np.full(3 * nele, np.pi)])
    mv_frac = opt.move_frac if opt.optimize_density else 0.0
    move_z = np.full(nele, opt.move_z)
    move_frac = np.full(nele, mv_frac)
    if passive_z.any():
        move_z[passive_z] = 0.0
        if opt.passive_frac_value is not None:
            move_frac[passive_z] = 0.0
    move_vector = np.concatenate([move_z, move_frac, np.full(3 * nele, opt.move_angle)])

    updater_cls = SIGradientUpdater if opt.si_schedule else ALGradientUpdater
    updater = updater_cls(lower, upper)
    angle_updater = updater_cls(
        np.full(3 * nele, -np.pi), np.full(3 * nele, np.pi)
    )
    angle_mv = np.full(3 * nele, opt.move_angle)
    angle_phase_end = (int(opt.angle_phase_iter) if opt.angle_phase_iter is not None
                       else int(opt.angle_phase_frac * opt.max_iter))
    p0 = opt.penal_steps[0] if opt.penal_steps else opt.penal
    params = SpinodalParams(penal=p0, beta=opt.beta0, eta=opt.eta, Emin=opt.Emin)

    # SIMP p-continuation (paper: p=[1,1.5,2,2.5,3] over [150,100,100,50,50] iters);
    # Heaviside beta ramps only after continuation completes. With cont_tol > 0
    # (SI S4.1: tol=0.02) a continuation step also completes early once the
    # selection variables stop changing (max|dz| < tol).
    cont_k = 0
    cont_iters = 0
    cont_done = not opt.penal_steps
    beta_ramp_start = None if opt.penal_steps else opt.beta_start_iter
    z_change = np.inf
    # The SI does not say whether tol=0.02 must hold for a single iteration or
    # sustainedly; a single quiet iteration is too noisy (the compliance
    # gradient scale drops as c falls, which can stall z while the volume
    # constraint is still far from satisfied: the AL subproblem is stationary
    # for the CURRENT small lambda/mu, yet the design is deeply infeasible,
    # and advancing p then locks in a poor local minimum -- observed
    # empirically). Early advance therefore requires tol to hold over a
    # 5-iteration window (the AL update period) AND near-feasibility g < 0.01.
    # The MaxIter-based advance stays unconditional, as in the SI.
    z_quiet = 0
    g_last = np.inf

    hist = {"c": [], "vol": [], "g": [], "change": [], "beta": [], "penal": []}
    obj0 = None
    last = None

    for it in range(opt.max_iter):
        if opt.penal_steps:
            if not cont_done and (
                    cont_iters >= opt.penal_iters[cont_k]
                    or (opt.cont_tol > 0 and z_quiet >= 5 and g_last < 1e-2)):
                cont_k += 1
                cont_iters = 0
                z_quiet = 0
                if cont_k >= len(opt.penal_steps):
                    cont_done = True
                    beta_ramp_start = it
                    if opt.verbose:
                        print(f"  [cont] p-continuation done at it {it}; beta ramp begins")
                elif opt.verbose:
                    print(f"  [cont] step {cont_k} (p={opt.penal_steps[cont_k]}) at it {it}")
            params.penal = opt.penal_steps[min(cont_k, len(opt.penal_steps) - 1)]
        # Heaviside continuation (additive per paper if beta_add>0, else multiplicative)
        if (beta_ramp_start is not None and it > beta_ramp_start
                and (it - beta_ramp_start) % opt.beta_period == 0):
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
        g_last = g
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

        z_change = float(np.max(np.abs(x_new[:nele] - x_old[:nele])))
        z_quiet = z_quiet + 1 if z_change < opt.cont_tol else 0
        cont_iters += 1

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
