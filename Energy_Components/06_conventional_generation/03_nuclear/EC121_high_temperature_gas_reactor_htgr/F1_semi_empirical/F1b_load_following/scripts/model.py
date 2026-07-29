"""
EC121 -- High Temperature Gas Reactor (HTGR) -- F1b Load-Following Model

Extends HTGR power map with:
  1. Xe-135 / I-135 dynamics -- same as PWR but with graphite thermal mass
     moderating the transient response (thermal time constant ~hours).

  2. Graphite thermal mass inertia:
     The large graphite moderator heat capacity (~280 MJ/K for 250 MWth module)
     slows fuel temperature changes during load-following, effectively damping
     rapid power transients. During a power ramp, the fuel temperature lags
     behind the steady-state value by a time constant tau.
     T_fuel(t) = T_fuel_ss + (T_fuel_0 - T_fuel_ss) * exp(-t / tau_min)

  3. Temperature reactivity feedback (strongly negative):
     rho_temp = temp_coeff * delta_T_fuel_delayed
     The large negative temperature coefficient of TRISO fuel provides
     inherent power self-regulation and passive safety.

  4. Ramp rate limited by:
     - Graphite thermal mass (slow temperature equilibration)
     - Helium circulator response (positive flow margin needed for safety)
     - ~7%/min typical for pebble-bed design

Key HTGR vs PWR differences:
  - Much larger thermal inertia from graphite moderator (280 MJ/K vs ~100 MJ/K water)
  - Higher operating temperature (750-950C helium) enables more efficient conversion
  - Stronger negative temperature coefficient (Doppler + graphite spectrum shift)
  - No meltdown risk: TRISO fuel retains fission products up to 1600C

References:
    Dong, Y. (2011). Fuel management of the pebble bed reactor.
        Nucl. Eng. Des., 241(12), 4755-4762.
    Zhang, Z. et al. (2009). The design of Chinese pebble-bed reactor HTR-PM.
        Nucl. Eng. Des., 239(7), 2265-2274.
    Muto, Y. et al. (2003). Performance calculations for GTHTR300 gas turbine.
        IAEA-TECDOC-1318.
    Reitsma, F. (2004). IAEA Coordinate Research Project on HTGRs.
        IAEA-TECDOC-1528.
"""

import numpy as np


class HTGRF1b:
    """HTGR pebble-bed load-following model with graphite thermal inertia and Xe dynamics."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_thermal        = u["P_thermal_mw"]["value"]
        self.eta              = u["eta_thermal"]["value"]
        self.sigma_Xe         = u["sigma_Xe_cm2"]["value"]
        self.lambda_Xe        = u["lambda_Xe_per_s"]["value"]
        self.lambda_I         = u["lambda_I_per_s"]["value"]
        self.gamma_I          = u["gamma_I"]["value"]
        self.gamma_Xe         = u["gamma_Xe"]["value"]
        self.ramp_limit       = u["ramp_rate_limit_pct_min"]["value"]
        self.PLR_min          = u["PLR_min"]["value"]
        self.PLR_max          = u["PLR_max"]["value"]
        self.total_margin     = u["total_reactivity_margin_pcm"]["value"]
        self.xe_react_coeff   = u["xenon_reactivity_coeff_pcm"]["value"]
        self.graphite_mass_mj = u["graphite_thermal_mass_mj_per_k"]["value"]  # MJ/K
        self.temp_coeff       = u["temp_coeff_pcm_per_K"]["value"]             # pcm/K (negative)
        self.T_He_inlet       = u["T_helium_inlet_C"]["value"]
        self.T_He_outlet      = u["T_helium_outlet_C"]["value"]

        # Thermal time constant: tau = C / P  (very rough: how long to heat/cool graphite)
        # C [MJ/K], P [MW] → tau [s]; use rated power for reference
        self.tau_s = self.graphite_mass_mj / self.P_thermal * 1e6 / 1e6  # s
        # = graphite_mass_mj [MJ/K] / P_thermal [MW] = seconds/K * K = seconds? No:
        # Actually: tau [s] = C [J/K] / (dQ/dT) but here we use dimensional estimate
        # tau = C / P_rated = (280 MJ/K) / (250 MW) * [K_change_per_PLR_change]
        # Simplification: use tau = 280/250 hours = 1.12 hours
        self.tau_s = (self.graphite_mass_mj / self.P_thermal) * 3600.0  # s (~4032 s for 280/250)

    # ------------------------------------------------------------------
    # Xenon dynamics (same as PWR; graphite is thermal moderator)
    # ------------------------------------------------------------------

    def equilibrium_xenon(self, power_fraction):
        """Normalized equilibrium Xe-135 at given power fraction."""
        phi = np.clip(np.asarray(power_fraction, dtype=float), 0.0, 1.0)
        gamma_total = self.gamma_I + self.gamma_Xe
        Xe_eq = gamma_total * phi / (self.lambda_Xe + self.sigma_Xe * phi)
        Xe_eq_full = gamma_total * 1.0 / (self.lambda_Xe + self.sigma_Xe * 1.0)
        if Xe_eq_full > 0:
            Xe_eq = Xe_eq / Xe_eq_full
        return Xe_eq

    def xenon_transient(self, previous_power_fraction, new_power_fraction,
                        time_hours):
        """
        Xe-135 transient after a power change.
        Same analytical two-exponential model as PWR F1b.
        The graphite thermal mass affects power dynamics (temperature feedback),
        but I/Xe nuclear kinetics are unchanged.
        """
        P1 = np.clip(float(previous_power_fraction), 0.0, 1.0)
        P2 = np.clip(float(new_power_fraction), 0.0, 1.0)
        t  = float(time_hours) * 3600.0

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
    # Graphite thermal mass inertia
    # ------------------------------------------------------------------

    def fuel_temp_ss_C(self, power_fraction):
        """
        Steady-state fuel temperature at given PLR.
        Approximate: T_fuel_avg = T_inlet + (T_outlet - T_inlet) * PLR
        (outlet temperature scales with power fraction)
        """
        plr = np.clip(np.asarray(power_fraction, dtype=float), 0.0, 1.0)
        return self.T_He_inlet + (self.T_He_outlet - self.T_He_inlet) * plr

    def fuel_temp_transient_C(self, previous_plr, current_plr, time_hours):
        """
        Actual fuel/graphite temperature after transition, accounting for thermal inertia.

        T_fuel(t) = T_ss + (T_0 - T_ss) * exp(-t / tau)
        where tau = graphite_mass / rated_power [hours]

        This first-order lag model represents the graphite thermal mass
        slowing down temperature changes during load-following.
        """
        T_ss = float(self.fuel_temp_ss_C(current_plr))
        T_0  = float(self.fuel_temp_ss_C(previous_plr))
        tau_h = self.tau_s / 3600.0   # hours
        t_h   = float(time_hours)

        if tau_h < 1e-6:
            return T_ss
        T_t = T_ss + (T_0 - T_ss) * np.exp(-t_h / tau_h)
        return float(T_t)

    # ------------------------------------------------------------------
    # Temperature reactivity feedback
    # ------------------------------------------------------------------

    def temperature_reactivity_pcm(self, T_current_C, T_rated_C=None):
        """
        Temperature reactivity feedback [pcm].
        rho_T = temp_coeff * (T_current - T_rated)
        At part load (T drops), negative coeff * negative dT → positive reactivity.
        """
        if T_rated_C is None:
            T_rated_C = float(self.fuel_temp_ss_C(1.0))
        delta_T = float(T_current_C) - float(T_rated_C)
        return float(self.temp_coeff) * delta_T

    # ------------------------------------------------------------------
    # Reactivity balance
    # ------------------------------------------------------------------

    def xenon_reactivity_pcm(self, xe_relative):
        """Reactivity penalty from Xe [pcm]."""
        return self.xe_react_coeff * xe_relative

    def available_reactivity_pcm(self, xe_relative, T_current_C):
        """
        Available reactivity = total margin + Xe penalty + temperature feedback.
        """
        rho_Xe = self.xenon_reactivity_pcm(xe_relative)
        rho_T  = self.temperature_reactivity_pcm(T_current_C)
        return self.total_margin + rho_Xe + rho_T

    def can_restart(self, xe_relative, T_current_C):
        """Whether reactor can achieve target power given Xe and temperature state."""
        return self.available_reactivity_pcm(xe_relative, T_current_C) > 0

    # ------------------------------------------------------------------
    # Ramp rate
    # ------------------------------------------------------------------

    def ramp_rate_limit(self, current_power, target_power, time_minutes):
        """Achievable power change within HTGR ramp rate limit (7%/min)."""
        current    = float(current_power)
        target     = float(target_power)
        dt_min     = float(time_minutes)
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
        Compute HTGR load-following state.

        Returns dict with:
            power_output_mw           : Electrical output [MW_e]
            xenon_concentration_rel   : Xe relative to full-power equilibrium
            fuel_temp_C               : Actual fuel/graphite temperature [C] (delayed by inertia)
            temperature_reactivity_pcm: Temperature feedback [pcm]
            available_reactivity_pcm  : Net available reactivity [pcm]
            ramp_rate_limit_pct_min   : Maximum ramp rate [%/min]
            can_restart               : Whether restart is possible
        """
        PLR    = np.clip(float(power_fraction), self.PLR_min, self.PLR_max)
        t_h    = float(time_at_power_hours)
        P_prev = np.clip(float(previous_power_fraction), 0.0, 1.0)

        P_electric = self.P_thermal * PLR * self.eta
        Xe_rel     = self.xenon_transient(P_prev, PLR, t_h)
        T_fuel     = self.fuel_temp_transient_C(P_prev, PLR, t_h)
        rho_T      = self.temperature_reactivity_pcm(T_fuel)
        avail_pcm  = self.available_reactivity_pcm(Xe_rel, T_fuel)
        restart_ok = self.can_restart(Xe_rel, T_fuel)

        return {
            "power_output_mw":            float(P_electric),
            "xenon_concentration_rel":    float(Xe_rel),
            "fuel_temp_C":                float(T_fuel),
            "temperature_reactivity_pcm": float(rho_T),
            "available_reactivity_pcm":   float(avail_pcm),
            "ramp_rate_limit_pct_min":    float(self.ramp_limit),
            "can_restart":                bool(restart_ok),
        }
