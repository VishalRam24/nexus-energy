"""
EC119 -- Molten Salt Reactor (MSR) -- F1b Load-Following Model

Extends the MSR power map with:
  1. Xe-135 / I-135 dynamics -- same fission product kinetics as PWR F1b,
     BUT modified by online Xe stripping (off-gas system removes Xe from
     liquid salt continuously).

  2. Effective Xe removal: lambda_eff_Xe = lambda_Xe + lambda_strip
     where lambda_strip = xe_strip_efficiency * lambda_Xe
     This reduces the peak Xe buildup after a power reduction and eliminates
     the "xenon deadtime" (xenon preclude) problem common to solid-fuel reactors.

  3. Fuel salt temperature reactivity feedback (inherent safety):
     rho_temp = fuel_temp_coeff * delta_T_fuel
     where delta_T_fuel = T_rated * (1 - PLR)  [approx, from power balance]
     Large negative temperature coefficient provides passive power self-regulation.

  4. Faster ramp rates (~10%/min) due to:
     - Liquid fuel (no pellet thermal stress)
     - Online Xe removal eliminates xenon constraints
     - No steam generator thermal mass (direct cycle or intermediate loop)

Key MSR vs PWR differences:
  - Xe stripping reduces xenon worth to ~50% of PWR value
  - Xenon deadtime effectively eliminated (can restart promptly after shutdown)
  - Much higher PLR range (20-100%) enabled by liquid fuel flexibility
  - Strongly negative fuel temperature coefficient provides self-regulation

References:
    Haubenreich, P.N. & Engel, J.R. (1970). Experience with the Molten-Salt
        Reactor Experiment. Nucl. Appl. Technol., 8(2), 118-136.
    MacPherson, H.G. (1985). The Molten Salt Reactor Adventure.
        Nucl. Sci. Eng., 90, 374-380.
    Delpech, S. et al. (2009). Reactor physic and reprocessing scheme for
        innovative molten salt reactor. J. Fluorine Chem., 130(1), 11-17.
    NEA (2021). Molten Salt Reactor Technology Handbook.
"""

import numpy as np


class MSRF1b:
    """MSR load-following model with Xe dynamics modified by online Xe stripping."""

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
        self.xe_strip_eff    = u["xe_strip_efficiency"]["value"]
        self.fuel_temp_coeff = u["fuel_temp_coeff_pcm_per_K"]["value"]   # pcm/K (negative)
        self.T_fuel_rated    = u["T_fuel_rated_C"]["value"]              # degC

        # Effective Xe decay constant including online stripping
        # Off-gas system removes Xe at rate proportional to lambda_Xe
        self.lambda_Xe_eff = self.lambda_Xe * (1.0 + self.xe_strip_eff)

    # ------------------------------------------------------------------
    # Xenon dynamics (with stripping)
    # ------------------------------------------------------------------

    def equilibrium_xenon(self, power_fraction):
        """
        Normalized equilibrium Xe-135 at given power level.
        In MSR: Xe is continuously removed by off-gas, so effective lambda_Xe_eff
        is higher → equilibrium Xe is lower than in solid-fuel reactors.
        Normalized to 1.0 at full power equilibrium.
        """
        phi = np.clip(np.asarray(power_fraction, dtype=float), 0.0, 1.0)
        gamma_total = self.gamma_I + self.gamma_Xe

        # Equilibrium uses effective lambda (includes stripping)
        Xe_eq = gamma_total * phi / (self.lambda_Xe_eff + self.sigma_Xe * phi)
        Xe_eq_full = gamma_total * 1.0 / (self.lambda_Xe_eff + self.sigma_Xe * 1.0)

        if Xe_eq_full > 0:
            Xe_eq = Xe_eq / Xe_eq_full
        return Xe_eq

    def xenon_transient(self, previous_power_fraction, new_power_fraction,
                        time_hours):
        """
        Xe-135 concentration after a power change (with online stripping).

        The off-gas system continuously removes Xe from the circulating salt,
        so the effective xenon decay constant is higher:
            lambda_eff_Xe = lambda_Xe + lambda_strip
        This dramatically reduces the xenon overshoot after power reduction.

        Returns Xe_relative: normalized to full-power equilibrium.
        """
        P1 = np.clip(float(previous_power_fraction), 0.0, 1.0)
        P2 = np.clip(float(new_power_fraction), 0.0, 1.0)
        t  = float(time_hours) * 3600.0  # s

        Xe_eq_1 = float(self.equilibrium_xenon(P1))
        Xe_eq_2 = float(self.equilibrium_xenon(P2))

        # Effective destruction rate at new power: decay + stripping + neutron capture
        lambda_eff = self.lambda_Xe_eff + self.sigma_Xe * P2

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
    # Temperature reactivity feedback
    # ------------------------------------------------------------------

    def fuel_temperature_C(self, power_fraction):
        """
        Approximate fuel salt average temperature at given power fraction.
        Linear: T_fuel = T_rated * PLR  (simplified; actual depends on flow)
        Returns delta T relative to rated.
        """
        plr = np.clip(np.asarray(power_fraction, dtype=float), 0.0, 1.0)
        return self.T_fuel_rated * plr

    def temperature_reactivity_pcm(self, power_fraction):
        """
        Reactivity from fuel salt temperature change relative to rated.
        rho_T = fuel_temp_coeff [pcm/K] * (T_fuel(PLR) - T_fuel_rated)
        At PLR < 1: T_fuel drops → positive reactivity insertion (negative coeff * negative dT).
        """
        T_fuel = float(np.atleast_1d(self.fuel_temperature_C(power_fraction))[0])
        delta_T = T_fuel - self.T_fuel_rated
        return float(self.fuel_temp_coeff) * delta_T

    # ------------------------------------------------------------------
    # Reactivity balance
    # ------------------------------------------------------------------

    def xenon_reactivity_pcm(self, xe_relative):
        """Reactivity penalty from Xe [pcm]; negative = poison."""
        return self.xe_react_coeff * xe_relative

    def available_reactivity_pcm(self, xe_relative, power_fraction):
        """
        Available reactivity = total margin + Xe penalty + temperature feedback.
        If < 0, reactor cannot maintain criticality at desired power.
        """
        rho_Xe  = self.xenon_reactivity_pcm(xe_relative)
        rho_T   = self.temperature_reactivity_pcm(power_fraction)
        return self.total_margin + rho_Xe + rho_T

    def can_restart(self, xe_relative, power_fraction=1.0):
        """MSR restart: very rarely blocked due to Xe stripping + lower Xe worth."""
        return self.available_reactivity_pcm(xe_relative, power_fraction) > 0

    # ------------------------------------------------------------------
    # Ramp rate
    # ------------------------------------------------------------------

    def ramp_rate_limit(self, current_power, target_power, time_minutes):
        """Achievable power change within ramp rate limit. Returns (achievable, was_limited)."""
        current   = float(current_power)
        target    = float(target_power)
        dt_min    = float(time_minutes)
        max_change = self.ramp_limit / 100.0 * dt_min

        if abs(target - current) <= max_change:
            return target, False
        elif target > current:
            return current + max_change, True
        else:
            return current - max_change, True

    # ------------------------------------------------------------------
    # Main predict
    # ------------------------------------------------------------------

    def predict(self, power_fraction, time_at_power_hours, previous_power_fraction):
        """
        Compute MSR load-following state.

        Returns dict with:
            power_output_mw           : Electrical output [MW_e]
            xenon_concentration_rel   : Xe relative to full-power equilibrium
            temperature_reactivity_pcm: Fuel salt temperature feedback [pcm]
            available_reactivity_pcm  : Available reactivity [pcm]
            ramp_rate_limit_pct_min   : Maximum ramp rate [%/min]
            can_restart               : Whether restart is possible
        """
        PLR    = np.clip(float(power_fraction), self.PLR_min, self.PLR_max)
        t_h    = float(time_at_power_hours)
        P_prev = np.clip(float(previous_power_fraction), 0.0, 1.0)

        P_electric  = self.P_thermal * PLR * self.eta
        Xe_rel      = self.xenon_transient(P_prev, PLR, t_h)
        rho_T       = self.temperature_reactivity_pcm(PLR)
        avail_pcm   = self.available_reactivity_pcm(Xe_rel, PLR)
        restart_ok  = self.can_restart(Xe_rel, PLR)

        return {
            "power_output_mw":            float(P_electric),
            "xenon_concentration_rel":    float(Xe_rel),
            "temperature_reactivity_pcm": float(rho_T),
            "available_reactivity_pcm":   float(avail_pcm),
            "ramp_rate_limit_pct_min":    float(self.ramp_limit),
            "can_restart":                bool(restart_ok),
        }
