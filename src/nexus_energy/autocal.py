"""N_En_Phase 20 — auto-calibration: fit model parameters from telemetry.

Generalizes the scalar CO₂-price recovery (``diff_bridge.fit_co2_price``)
to an m-dimensional parameter vector with a damped Gauss-Newton driver,
then wraps it in a moving-horizon :class:`AutoCalibrator` so a running
MPC keeps its own model honest — no human intervention:

    each cycle:  MPC plans with believed params
                 → plant telemetry arrives
                 → AutoCalibrator.step(window) corrects the beliefs
                 → next plan uses the corrected model.

The two design rules that make this trustworthy in operations:

* **Identifiability gate** — a parameter whose Jacobian column carries
  (almost) no signal in this window ("the battery barely cycled") is
  FROZEN for the cycle and flagged ``data_silent``, never nudged by
  noise. The gradient J ≈ 0 IS the statement "this data cannot identify
  the parameter here" — the same honesty that produced the Belgium
  identified-set result (N_En_Phase 19.C).
* **Slew limit** (``max_rel_step``) — beliefs move at most a bounded
  relative amount per cycle, so one bad window cannot wreck the model.

Pure numpy; analytic Jacobians come from the diff layer
(:mod:`nexus_energy.diff`) via a caller-supplied ``jacobian_fn`` —
any parameter that is chain-rulable from the exposed blocks works
(marginal costs, fuel-price/efficiency via mc = fuel/η, CO₂ price via
emissions, storage charge/discharge efficiency via the η blocks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class CalibrationReport:
    """Outcome of one :func:`fit_params` run / one AutoCalibrator cycle."""
    params_old: dict
    params_new: dict
    frozen: dict            # name → True when data-silent this window
    loss_history: list      # accepted-step losses
    n_solves: int
    converged: bool
    message: str = ""

    def changed(self, rel_tol: float = 1e-9) -> dict:
        """Parameters that actually moved (name → (old, new))."""
        out = {}
        for k, old in self.params_old.items():
            new = self.params_new[k]
            if abs(new - old) > rel_tol * max(abs(old), 1.0):
                out[k] = (old, new)
        return out


def fit_params(
    make_solution: Callable[[dict], object],
    observed: np.ndarray,
    params: dict,
    jacobian_fn: Callable[[object, dict], np.ndarray],
    *,
    residual_fn: Optional[Callable[[object], np.ndarray]] = None,
    n_iter: int = 30,
    tol: float = 1e-8,
    lm_damping: float = 1e-6,
    lm_growth: float = 10.0,
    lm_shrink: float = 0.3,
    lm_max_retries: int = 8,
    gate_threshold: float = 1e-3,
    max_rel_step: Optional[float] = None,
    verbose: bool = False,
) -> CalibrationReport:
    """Damped multi-parameter Gauss-Newton with identifiability gating.

    Minimises ``½‖r(θ)‖²`` where, by default,
    ``r = solution.dispatch − observed``; pass ``residual_fn`` to fit a
    different observable (shares, SOC traces, …).

    Args:
        make_solution: θ-dict → SOLVED diff-layer solution (must carry the
            Jacobian blocks ``jacobian_fn`` needs). One model solve per call.
        observed: observation array; flattened against the residual.
        params: name → (value, lo, hi) initial beliefs + box bounds.
            Insertion order defines the Jacobian column order.
        jacobian_fn: (solution, θ) → J of shape (len(r), m) — analytic
            chain-rule columns, NO extra solves.
        gate_threshold: freeze parameter i this run when
            ``‖J_i‖² < gate_threshold · ‖J‖²_F / m`` (relative column-energy
            test — scale-free).
        max_rel_step: per-run trust region: |δ_i| ≤ max_rel_step·max(|θ_i|,1e-6).

    Scaled Levenberg-Marquardt: ``(JᵀJ + λ·diag(JᵀJ))δ = −Jᵀr`` with
    accept/reject loss control — flat pieces (piecewise-linear dispatch)
    inflate λ instead of exploding the step.
    """
    names = list(params.keys())
    m = len(names)
    theta = np.array([float(params[k][0]) for k in names])
    lo = np.array([float(params[k][1]) for k in names])
    hi = np.array([float(params[k][2]) for k in names])
    obs = np.asarray(observed, dtype=float).reshape(-1)

    def theta_dict(vec):
        return {k: float(v) for k, v in zip(names, vec)}

    def resid(sol):
        if residual_fn is not None:
            return np.asarray(residual_fn(sol), dtype=float).reshape(-1)
        return (np.asarray(sol.dispatch, dtype=float).reshape(-1) - obs)

    frozen = {k: False for k in names}
    loss_hist: list = []
    n_solves = 0
    lam = float(lm_damping)
    converged = False
    message = ""

    sol = make_solution(theta_dict(theta))
    n_solves += 1
    r = resid(sol)
    loss = 0.5 * float(r @ r)
    loss_hist.append(loss)

    for it in range(n_iter):
        J = np.asarray(jacobian_fn(sol, theta_dict(theta)), dtype=float)
        if J.shape != (r.size, m):
            raise ValueError(f"jacobian_fn returned {J.shape}, "
                             f"expected {(r.size, m)}")
        # Scale-aware identifiability: compare the residual response to a
        # RELATIVE change of each parameter (‖J_i‖²·θ_i²), not raw column
        # norms — mixed units ($/MWh vs unitless η) would otherwise let
        # large-Jacobian parameters silence small-Jacobian ones.
        col_sq = (J * J).sum(axis=0)
        col_energy = col_sq * np.maximum(np.abs(theta), 1.0) ** 2
        total = float(col_energy.sum())
        gate = gate_threshold * total / m if total > 0 else np.inf
        active = col_energy >= gate
        for i, k in enumerate(names):
            frozen[k] = not bool(active[i])
        if not active.any():
            converged = loss <= 1e-18
            message = "all parameters data-silent in this window"
            break

        Ja = J[:, active]
        g = Ja.T @ r
        JtJ = Ja.T @ Ja
        scale = np.maximum(np.diag(JtJ), 1e-12)

        accepted = False
        for _retry in range(lm_max_retries):
            try:
                delta_a = np.linalg.solve(JtJ + lam * np.diag(scale), -g)
            except np.linalg.LinAlgError:
                delta_a = np.linalg.lstsq(
                    JtJ + lam * np.diag(scale), -g, rcond=None)[0]
            delta = np.zeros(m)
            delta[active] = delta_a
            if max_rel_step is not None:
                cap = max_rel_step * np.maximum(np.abs(theta), 1e-6)
                delta = np.clip(delta, -cap, cap)
            theta_try = np.clip(theta + delta, lo, hi)
            sol_try = make_solution(theta_dict(theta_try))
            n_solves += 1
            r_try = resid(sol_try)
            loss_try = 0.5 * float(r_try @ r_try)
            if loss_try < loss or loss <= 1e-18:
                theta, sol, r, loss = theta_try, sol_try, r_try, loss_try
                loss_hist.append(loss)
                lam = max(lam * lm_shrink, 1e-12)
                accepted = True
                break
            lam *= lm_growth
        if verbose:
            print(f"  it={it:02d} loss={loss:.6e} lam={lam:.1e} "
                  f"theta={theta_dict(theta)}")
        if not accepted:
            message = "LM stalled (flat piece / local kink)"
            break
        step_inf = float(np.max(np.abs(delta)))
        if step_inf < tol or loss <= 1e-18:
            converged = True
            break

    return CalibrationReport(
        params_old={k: float(params[k][0]) for k in names},
        params_new=theta_dict(theta),
        frozen=frozen,
        loss_history=loss_hist,
        n_solves=n_solves,
        converged=converged,
        message=message,
    )


@dataclass
class AutoCalibrator:
    """Moving-horizon parameter tracker for a running MPC.

    Holds the current believed parameters and, every ``update_every``-th
    call to :meth:`step`, refits them against the latest telemetry window
    with :func:`fit_params` (seeded at the current beliefs, slew-limited
    by ``max_rel_step``). Returns the :class:`CalibrationReport` (or
    ``None`` on skipped cycles); the updated beliefs are at
    :attr:`params`.

    ``solution_factory(window, θ)`` must return a solved diff-layer
    solution of the believed model under that window's exogenous data;
    ``jacobian_fn(solution, θ)`` provides the analytic columns;
    ``observed_fn(window)`` extracts the telemetry the residual targets.
    """
    solution_factory: Callable
    jacobian_fn: Callable
    observed_fn: Callable
    params: dict                      # name → (value, lo, hi)
    enabled: bool = True              # master toggle — see enable()/disable()
    update_every: int = 1
    n_iter_per_cycle: int = 8
    lm_damping: float = 1e-4
    gate_threshold: float = 1e-3
    max_rel_step: float = 0.25
    residual_fn: Optional[Callable] = None
    verbose: bool = False
    # ---- noise-aware MHE controls (20.x.2) ----
    # noise_std: expected per-observation telemetry noise σ. When > 0:
    #   * smooth (EMA gain α ∈ (0,1]) averages fitted values across
    #     windows instead of jumping to each window's noisy fit;
    #   * a window whose post-fit loss exceeds outlier_zscore²·(½nσ²) is
    #     REJECTED outright (sensor fault / unmodeled event) — beliefs
    #     stay put and the report says so.
    noise_std: float = 0.0
    smooth: float = 1.0
    outlier_zscore: float = 4.0
    reports: list = field(default_factory=list)
    locked: set = field(default_factory=set)   # per-param manual locks
    _cycle: int = 0

    # ---- operator controls (toggle / manual override) ----------------
    def enable(self) -> None:
        """Resume automatic calibration (master switch)."""
        self.enabled = True

    def disable(self) -> None:
        """Suspend automatic calibration; beliefs stay exactly as-is.
        ``step()`` still counts cycles but performs no fits."""
        self.enabled = False

    def lock_param(self, name: str) -> None:
        """Exclude one parameter from auto-updates (operator override
        stays in force until :meth:`unlock_param`)."""
        if name not in self.params:
            raise KeyError(name)
        self.locked.add(name)

    def unlock_param(self, name: str) -> None:
        self.locked.discard(name)

    def set_param(self, name: str, value: float, *, lock: bool = False) -> None:
        """Manually set a believed value (clamped to its box bounds).
        ``lock=True`` also locks it against future auto-updates."""
        if name not in self.params:
            raise KeyError(name)
        _, lo, hi = self.params[name]
        self.params[name] = (float(np.clip(value, lo, hi)), lo, hi)
        if lock:
            self.locked.add(name)

    def step(self, window) -> Optional[CalibrationReport]:
        self._cycle += 1
        if not self.enabled:
            return None
        if (self._cycle - 1) % self.update_every != 0:
            return None
        if set(self.params) <= self.locked:
            return None  # everything operator-locked: nothing to fit
        # Locked params enter the fit with degenerate bounds (lo=hi=value)
        # so the optimizer holds them exactly while fitting the rest.
        fit_param_spec = {
            k: ((v, v, v) if k in self.locked else (v, lo, hi))
            for k, (v, lo, hi) in self.params.items()
        }
        report = fit_params(
            lambda th: self.solution_factory(window, th),
            self.observed_fn(window),
            fit_param_spec,
            self.jacobian_fn,
            residual_fn=self.residual_fn,
            n_iter=self.n_iter_per_cycle,
            lm_damping=self.lm_damping,
            gate_threshold=self.gate_threshold,
            max_rel_step=self.max_rel_step,
            verbose=self.verbose,
        )
        # ---- noise-aware acceptance (20.x.2) ----
        if self.noise_std > 0:
            n_obs = np.asarray(self.observed_fn(window)).size
            expected_loss = 0.5 * n_obs * self.noise_std ** 2
            if report.loss_history[-1] > \
                    self.outlier_zscore ** 2 * max(expected_loss, 1e-30):
                # The fitted model cannot explain this window even
                # approximately → sensor fault / unmodeled event. Do NOT
                # let it move the beliefs.
                for k in self.params:
                    report.params_new[k] = self.params[k][0]
                report.message = (
                    f"outlier window REJECTED (loss "
                    f"{report.loss_history[-1]:.3e} > "
                    f"{self.outlier_zscore}²×expected {expected_loss:.3e})")
                report.converged = False
                self.reports.append(report)
                return report

        # Commit with EMA smoothing (noise averaging across windows) and
        # the PER-CYCLE slew limit: regardless of how far the inner fit
        # moved, beliefs change at most max_rel_step per cycle — one bad
        # telemetry window cannot wreck the model. Operator-locked params
        # are never auto-updated.
        alpha = float(self.smooth) if self.noise_std > 0 else 1.0
        for k in self.params:
            old, lo, hi = self.params[k]
            if k in self.locked:
                report.params_new[k] = old
                report.frozen[k] = True  # reported as held (operator lock)
                continue
            blended = old + alpha * (report.params_new[k] - old)
            cap = self.max_rel_step * max(abs(old), 1e-6)
            new = float(np.clip(blended, old - cap, old + cap))
            new = float(np.clip(new, lo, hi))
            report.params_new[k] = new
            self.params[k] = (new, lo, hi)
        self.reports.append(report)
        return report

    @property
    def believed(self) -> dict:
        """Current point beliefs (name → value)."""
        return {k: v[0] for k, v in self.params.items()}
