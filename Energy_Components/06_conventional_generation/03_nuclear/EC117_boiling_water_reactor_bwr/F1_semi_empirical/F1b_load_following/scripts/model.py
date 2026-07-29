"""
EC117 -- Boiling Water Reactor (BWR) -- F1b Load-Following Model

Builds on F1a (steady-state power map) by adding:
  - Xenon-135 / Iodine-135 dynamics (poison transient) -- same physics as PWR
  - Ramp rate constraint: ~1%/min (slower than PWR due to two-phase flow
    stability and void fraction feedback requirements)
  - Void reactivity feedback signal (qualitative -- negative void coefficient
    is a key BWR safety characteristic)

Key BWR vs PWR differences:
  - Ramp rate limit: 1%/min (BWR) vs 5%/min (PWR)
    Reason: Direct cycle -- steam void fraction changes affect neutron flux
    directly without steam-generator thermal mass buffering.
  - PLR_min: 0.6 (BWR) vs 0.3 (PWR)
    Reason: Recirculation pump curve and two-phase flow stability constraints.
  - Slightly lower reactivity margin and Xe coefficient (harder neutron spectrum)

Xenon dynamics (same normalized model as PWR F1b):
    dI/dt = gamma_I * Sigma_f * phi - lambda_I * I
    dXe/dt = gamma_Xe * Sigma_f * phi + lambda_I * I - (lambda_Xe + sigma_Xe * phi) * Xe

Void feedback:
    Qualitative model: at part load, void fraction decreases (less boiling),
    providing positive reactivity feedback that the operator must compensate
    with control rods or recirculation flow reduction.
    rho_void = void_coeff * delta_void_pct
    (Modelled as a simple fractional change: delta_void ~ alpha * (1 - PLR))

References:
    Todreas, N.E. & Kazimi, M.S. (2012). Nuclear Systems, 2nd ed. CRC Press.
    NEA/CSNI (2011). BWR load-following operability.
    Duderstadt & Hamilton (1976). Nuclear Reactor Analysis. Wiley.
    March-Leuba, J. & Rey, J.M. (1993). Nucl. Eng. Des. 145, 97-111.
"""

import numpy as np


class BWRF1b:
    """BWR load-following model with xenon dynamics and void feedback."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_thermal       = u["P_thermal_mw"]["value"]
        self.eta             = u["eta_thermal"]["value"]
        self.sigma_Xe        = u["sigma_Xe_cm2"]["value"]
        self.lambda_Xe       = u["lambda_Xe_per_s"]["value"]
        self.lambda_I        = u["lambda_I_per_s"]["value"]
        self.gamma_I         = u["gamma_I"]["value"]
        self.gamma_Xe        = u["gamma_Xe"]["value"]
        self.ramp_limit      = u["ramp_rate_limit_pct_min"]["value"]
        self.PLR_min         = u["PLR_min"]["value"]
        self.PLR_max         = u["PLR_max"]["value"]
        self.total_margin    = u["total_reactivity_margin_pcm"]["value"]
        self.xe_react_coeff  = u["xenon_reactivity_coeff_pcm"]["value"]
        self.void_coeff      = u["void_coeff_pcm_per_pct_void"]["value"]

    # ------------------------------------------------------------------
    # Xenon dynamics (same formulation as PWR F1b)
    # ------------------------------------------------------------------

    def equilibrium_xenon(self, power_fraction):
        """
        Normalized equilibrium Xe-135 at given power level.
        Xe_eq(PLR) / Xe_eq(1.0) normalized to 1.0 at full power.
        """
        phi = np.clip(np.asarray(power_fraction, dtype=float), 0.0, 1.0)
        gamma_total = self.gamma_I + self.gamma_Xe
        Xe_eq = gamma_total * phi / (self.lambda_Xe + self.sigma_Xe * phi)
        Xe_eq_full = gamma_total * 1.0 / (self.lambda_Xe + self.sigma_Xe * 1.0)
        if Xe_eq_full > 0:
            Xe_eq = Xe_eq / Xe_eq_full
        return Xe_eq

    def xenon_transient(self, previous_power_fraction, new_power_fraction, time_hours):
        """
        Xe-135 concentration after a power change (normalized).
        Uses same two-exponential analytical approximation as PWR model.
        BWR physics are identical for I/Xe kinetics; differences in load-following
        come from the mechanical/hydraulic ramp rate constraints.

        Returns Xe_relative: Xe concentration relative to full-power equilibrium.
        """
        P1 = np.clip(float(previous_power_fraction), 0.0, 1.0)
        P2 = np.clip(float(new_power_fraction), 0.0, 1.0)
        t  = float(time_hours) * 3600.0  # convert to seconds

        Xe_eq_1 = float(self.equilibrium_xenon(P1))
        Xe_eq_2 = float(self.equilibrium_xenon(P2))

        lambda_eff = self.lambda_Xe + self.sigma_Xe * P2

        dI  = P1 - P2
        dXe = Xe_eq_1 - Xe_eq_2

        if abs(lambda_eff - self.lambda_I) > 1e-15:
            A1 = self.lambda_I * dI / (lambda_eff - self.lambda_I)
            A2 = dXe - A1
        else:
            A1 = 0.0
            A2 = dXe

        Xe_t = Xe_eq_2 + A1 * np.exp(-self.lambda_I * t) + A2 * np.exp(-lambda_eff * t)
        return max(0.0, float(Xe_t))

    # ------------------------------------------------------------------
    # Void feedback (BWR-specific)
    # ------------------------------------------------------------------

    def void_fraction_change_pct(self, current_plr, reference_plr=1.0):
        """
        Approximate change in average core void fraction relative to reference.
        At full power, void fraction ~ 40% in typical BWR.
        Linear approximation: delta_void ~ 20% * (reference_plr - current_plr)
        (i.e., void decreases at part load because boiling rate drops.)
        Returns: delta_void in percentage points (positive = more void)
        """
        # Simplification: void scales linearly with power fraction
        alpha = 20.0  # % void change from 0 to 100% PLR change
        return alpha * (float(current_plr) - float(reference_plr))

    def void_reactivity_pcm(self, current_plr, reference_plr=1.0):
        """
        Void feedback reactivity [pcm].
        Negative void coefficient: less void (lower PLR) -> positive reactivity insertion.
        rho_void = void_coeff [pcm/%void] * delta_void [%void]
        """
        delta_void = self.void_fraction_change_pct(current_plr, reference_plr)
        return self.void_coeff * delta_void

    # ------------------------------------------------------------------
    # Reactivity balance
    # ------------------------------------------------------------------

    def xenon_reactivity_pcm(self, xe_relative):
        """Reactivity penalty from xenon [pcm]."""
        return self.xe_react_coeff * xe_relative

    def available_reactivity_pcm(self, xe_relative, current_plr, reference_plr=1.0):
        """
        Available reactivity = total margin + Xe penalty + void feedback.
        If < 0, reactor cannot maintain criticality at desired power.
        """
        rho_Xe   = self.xenon_reactivity_pcm(xe_relative)
        rho_void = self.void_reactivity_pcm(current_plr, reference_plr)
        return self.total_margin + rho_Xe + rho_void

    def can_restart(self, xe_relative, current_plr=1.0):
        """Whether reactor can return to full power given current Xe level."""
        avail = self.available_reactivity_pcm(xe_relative, current_plr)
        return avail > 0

    # ------------------------------------------------------------------
    # Ramp rate
    # ------------------------------------------------------------------

    def ramp_rate_limit(self, current_power, target_power, time_minutes):
        """
        Achievable power within ramp rate limit.
        BWR limit: 1%/min (slower than PWR 5%/min).
        Returns: (achievable_power, was_limited)
        """
        current = float(current_power)
        target  = float(target_power)
        dt_min  = float(time_minutes)

        max_change = self.ramp_limit / 100.0 * dt_min  # fraction

        if abs(target - current) <= max_change:
            return target, False
        elif target > current:
            return current + max_change, True
        else:
            return current - max_change, True

    # ------------------------------------------------------------------
    # Main predict method
    # ------------------------------------------------------------------

    def predict(self, power_fraction, time_at_power_hours, previous_power_fraction):
        """
        Compute BWR load-following state.

        Args:
            power_fraction:          Current power level [PLR_min - 1.0]
            time_at_power_hours:     Time since last power change [hours]
            previous_power_fraction: Power level before the change [0-1.0]

        Returns:
            dict with:
                power_output_mw           : Electrical output [MW_e]
                xenon_concentration_rel   : Xe relative to full-power equilibrium
                available_reactivity_pcm  : Available reactivity [pcm]
                void_reactivity_pcm       : Void feedback reactivity [pcm]
                ramp_rate_limit_pct_min   : Maximum ramp rate [%/min]
                can_restart               : Whether return to full power is possible
        """
        PLR    = np.clip(float(power_fraction), self.PLR_min, self.PLR_max)
        t_h    = float(time_at_power_hours)
        P_prev = np.clip(float(previous_power_fraction), 0.0, 1.0)

        P_electric = self.P_thermal * PLR * self.eta

        Xe_rel     = self.xenon_transient(P_prev, PLR, t_h)
        rho_void   = self.void_reactivity_pcm(PLR)
        avail_pcm  = self.available_reactivity_pcm(Xe_rel, PLR)
        restart_ok = self.can_restart(Xe_rel, PLR)

        return {
            "power_output_mw":          float(P_electric),
            "xenon_concentration_rel":  float(Xe_rel),
            "available_reactivity_pcm": float(avail_pcm),
            "void_reactivity_pcm":      float(rho_void),
            "ramp_rate_limit_pct_min":  float(self.ramp_limit),
            "can_restart":              bool(restart_ok),
        }
