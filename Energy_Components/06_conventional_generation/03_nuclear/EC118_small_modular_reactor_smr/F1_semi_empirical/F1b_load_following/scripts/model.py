"""
EC118 -- Small Modular Reactor (SMR) -- F1b Load-Following + Thermal Inertia

Builds on F1a (steady-state power map) by adding:
  - Xenon-135 / Iodine-135 dynamics (same physics as PWR/BWR)
  - Ramp rate constraint: 5%/min (same as large PWR, but deeper range 20-100%)
  - Thermal inertia model: first-order lag on coolant temperature during ramp
    (integral design -> faster response than large loop-type PWR)
  - Achievable power during ramp (accounts for thermal time constant)

Key SMR vs Large PWR differences:
  - PLR_min: 0.2 (SMR) vs 0.3 (PWR)
    Reason: Designed specifically for deep load-following (grid firming, renewables integration)
  - Ramp rate: 5%/min (same capability, but over a wider range: 80% swing vs 70% for PWR)
  - Thermal time constant: ~8 min (SMR integral design, no separate steam generator)
    vs ~15-20 min for large 4-loop PWR
  - Smaller Xe worth (smaller core, lower neutron flux density per MWth)

Thermal inertia model:
    First-order lag (RC thermal circuit):
    T_coolant(t) = T_coolant_target + (T_coolant_initial - T_coolant_target) * exp(-t/tau)
    where tau = thermal_time_constant_min (minutes)

    T_outlet(PLR) = T_inlet + (T_outlet_rated - T_inlet) * PLR  [linear approximation]

    During a power ramp, the actual coolant temperature lags the steady-state target,
    which limits the effective thermal power delivered to the secondary.

References:
    IAEA (2022). Advances in Small Modular Reactor Technology Developments.
    NuScale Power Module Design Certification Documents (2020).
    Ingersoll, D.T. & Carelli, M.D. (2020). Handbook of Small Modular Nuclear Reactors.
    Todreas, N.E. & Kazimi, M.S. (2012). Nuclear Systems, 2nd ed. CRC Press.
"""

import numpy as np


class SMRF1b:
    """SMR load-following model: xenon dynamics + thermal inertia."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_thermal      = u["P_thermal_mw"]["value"]
        self.eta            = u["eta_thermal"]["value"]
        self.sigma_Xe       = u["sigma_Xe_cm2"]["value"]
        self.lambda_Xe      = u["lambda_Xe_per_s"]["value"]
        self.lambda_I       = u["lambda_I_per_s"]["value"]
        self.gamma_I        = u["gamma_I"]["value"]
        self.gamma_Xe       = u["gamma_Xe"]["value"]
        self.ramp_limit     = u["ramp_rate_limit_pct_min"]["value"]
        self.PLR_min        = u["PLR_min"]["value"]
        self.PLR_max        = u["PLR_max"]["value"]
        self.total_margin   = u["total_reactivity_margin_pcm"]["value"]
        self.xe_react_coeff = u["xenon_reactivity_coeff_pcm"]["value"]
        self.tau_min        = u["thermal_time_constant_min"]["value"]      # minutes
        self.primary_Cp     = u["primary_cp_mj_per_k"]["value"]            # MJ/K
        self.T_inlet        = u["T_inlet_c"]["value"]                      # degC
        self.T_outlet_rated = u["T_outlet_rated_c"]["value"]               # degC

    # ------------------------------------------------------------------
    # Xenon dynamics (same normalized model as PWR/BWR F1b)
    # ------------------------------------------------------------------

    def equilibrium_xenon(self, power_fraction):
        """Normalized equilibrium Xe-135 at given power level (1.0 at full power)."""
        phi = np.clip(np.asarray(power_fraction, dtype=float), 0.0, 1.0)
        gamma_total = self.gamma_I + self.gamma_Xe
        Xe_eq = gamma_total * phi / (self.lambda_Xe + self.sigma_Xe * phi)
        Xe_eq_full = gamma_total / (self.lambda_Xe + self.sigma_Xe)
        if Xe_eq_full > 0:
            Xe_eq = Xe_eq / Xe_eq_full
        return Xe_eq

    def xenon_transient(self, previous_power_fraction, new_power_fraction, time_hours):
        """
        Xe-135 concentration after a power change (normalized to full-power equilibrium).
        Two-exponential analytical solution.
        """
        P1 = np.clip(float(previous_power_fraction), 0.0, 1.0)
        P2 = np.clip(float(new_power_fraction), 0.0, 1.0)
        t  = float(time_hours) * 3600.0  # seconds

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
    # Thermal inertia model
    # ------------------------------------------------------------------

    def coolant_outlet_temp_steady(self, power_fraction):
        """
        Steady-state hot-leg coolant temperature at given PLR.
        Linear approximation: T_outlet = T_inlet + (T_rated - T_inlet) * PLR
        """
        PLR = np.clip(np.asarray(power_fraction, dtype=float), 0.0, 1.0)
        return self.T_inlet + (self.T_outlet_rated - self.T_inlet) * PLR

    def coolant_outlet_temp_transient(self, previous_power_fraction, new_power_fraction,
                                       time_minutes):
        """
        Actual coolant hot-leg temperature during a ramp (first-order lag).
        T(t) = T_target + (T_initial - T_target) * exp(-t / tau)
        where tau = thermal_time_constant_min [minutes].

        Returns actual coolant temperature [degC].
        """
        T_initial = float(self.coolant_outlet_temp_steady(previous_power_fraction))
        T_target  = float(self.coolant_outlet_temp_steady(new_power_fraction))
        t_min = float(time_minutes)
        tau   = self.tau_min  # minutes

        T_actual = T_target + (T_initial - T_target) * np.exp(-t_min / tau)
        return float(T_actual)

    def thermal_inertia_power_fraction(self, previous_power_fraction, new_power_fraction,
                                        time_minutes):
        """
        Effective thermal power fraction accounting for thermal lag.
        During a ramp, the secondary side receives power proportional to
        the actual temperature rise of the primary coolant, not the target.

        P_effective / P_rated = (T_outlet_actual - T_inlet) / (T_outlet_rated - T_inlet)
        Clipped to [0, PLR_max].
        """
        T_actual = self.coolant_outlet_temp_transient(
            previous_power_fraction, new_power_fraction, time_minutes)
        dT_actual = T_actual - self.T_inlet
        dT_rated  = self.T_outlet_rated - self.T_inlet
        return np.clip(dT_actual / dT_rated, 0.0, self.PLR_max)

    # ------------------------------------------------------------------
    # Reactivity
    # ------------------------------------------------------------------

    def xenon_reactivity_pcm(self, xe_relative):
        """Reactivity penalty from xenon [pcm]."""
        return self.xe_react_coeff * xe_relative

    def available_reactivity_pcm(self, xe_relative):
        """Available reactivity = total margin + Xe penalty."""
        return self.total_margin + self.xenon_reactivity_pcm(xe_relative)

    def can_restart(self, xe_relative):
        """Whether reactor can return to full power given current Xe level."""
        return self.available_reactivity_pcm(xe_relative) > 0

    # ------------------------------------------------------------------
    # Ramp rate
    # ------------------------------------------------------------------

    def ramp_rate_limit(self, current_power, target_power, time_minutes):
        """
        Achievable power within ramp rate limits (5%/min for SMR).
        Returns: (achievable_power, was_limited)
        """
        current = float(current_power)
        target  = float(target_power)
        dt_min  = float(time_minutes)

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

    def predict(self, power_fraction, time_at_power_hours, previous_power_fraction,
                time_since_ramp_start_minutes=None):
        """
        Compute SMR load-following state.

        Args:
            power_fraction:               Current target power level [0.2-1.0]
            time_at_power_hours:          Time since last power change (for Xe transient) [hours]
            previous_power_fraction:      Power level before the change [0-1.0]
            time_since_ramp_start_minutes: Time since ramp started (for thermal inertia) [min]
                                           If None, assumes steady-state (large t -> target reached)

        Returns dict with:
            power_output_mw              : Electrical output [MW_e]
            thermal_power_mw             : Reactor thermal power [MW_th]
            coolant_outlet_temp_c        : Hot-leg coolant temperature [degC]
            xenon_concentration_rel      : Xe relative to full-power equilibrium
            available_reactivity_pcm     : Available reactivity [pcm]
            ramp_rate_limit_pct_min      : Maximum ramp rate [%/min]
            can_restart                  : Whether restart to full power is possible
            thermal_lag_power_fraction   : Effective power fraction accounting for thermal lag
        """
        PLR    = np.clip(float(power_fraction), self.PLR_min, self.PLR_max)
        t_h    = float(time_at_power_hours)
        P_prev = np.clip(float(previous_power_fraction), 0.0, 1.0)

        # Thermal inertia: if ramp time not provided, assume steady-state (t >> tau)
        if time_since_ramp_start_minutes is None:
            t_ramp = 10.0 * self.tau_min  # large time -> steady state
        else:
            t_ramp = float(time_since_ramp_start_minutes)

        # Coolant temperature with lag
        T_outlet = self.coolant_outlet_temp_transient(P_prev, PLR, t_ramp)

        # Effective power fraction accounting for thermal inertia
        plr_thermal = self.thermal_inertia_power_fraction(P_prev, PLR, t_ramp)

        # Electrical output based on effective thermal power
        P_thermal = self.P_thermal * PLR          # neutron / fission power
        P_electric = P_thermal * plr_thermal * self.eta  # secondary receives lagged heat

        # Xenon
        Xe_rel    = self.xenon_transient(P_prev, PLR, t_h)
        avail_pcm = self.available_reactivity_pcm(Xe_rel)
        restart   = self.can_restart(Xe_rel)

        return {
            "power_output_mw":            float(P_electric),
            "thermal_power_mw":           float(P_thermal),
            "coolant_outlet_temp_c":      float(T_outlet),
            "xenon_concentration_rel":    float(Xe_rel),
            "available_reactivity_pcm":   float(avail_pcm),
            "ramp_rate_limit_pct_min":    float(self.ramp_limit),
            "can_restart":                bool(restart),
            "thermal_lag_power_fraction": float(plr_thermal),
        }
