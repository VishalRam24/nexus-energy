"""
EC116 -- Pressurized Water Reactor (PWR) -- F1b Load-Following Model

Builds on F1a (steady-state power map) by adding:
  - Simplified Xe-135 / I-135 dynamics (poison transients)
  - Ramp rate constraints (5%/min maximum)
  - Xenon-induced power oscillation and restart capability analysis

Xenon dynamics (normalized to equilibrium at full power):
    dI/dt = gamma_I * Sigma_f * phi - lambda_I * I
    dXe/dt = gamma_Xe * Sigma_f * phi + lambda_I * I - (lambda_Xe + sigma_Xe * phi) * Xe

where phi is proportional to power fraction.

After a power reduction:
  - Xe concentration peaks ~8-12 hours later (xenon overshoot)
  - If Xe reactivity exceeds available control rod margin, restart is blocked
  - This is the "xenon deadtime" or "xenon preclude" window

References:
    Todreas, N.E. & Kazimi, M.S. (2012). Nuclear Systems, 2nd ed. CRC Press.
    Stacey, W.M. (2007). Nuclear Reactor Physics, 2nd ed. Wiley.
    Duderstadt, J.J. & Hamilton, L.J. (1976). Nuclear Reactor Analysis. Wiley.
"""

import numpy as np


class PWRF1b:
    """PWR load-following model with xenon-135 dynamics."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_thermal = u["P_thermal_mw"]["value"]        # MW_th
        self.eta = u["eta_thermal"]["value"]                # -
        self.sigma_Xe = u["sigma_Xe_cm2"]["value"]          # cm2
        self.lambda_Xe = u["lambda_Xe_per_s"]["value"]      # 1/s
        self.lambda_I = u["lambda_I_per_s"]["value"]        # 1/s
        self.gamma_I = u["gamma_I"]["value"]                # -
        self.gamma_Xe = u["gamma_Xe"]["value"]              # -
        self.ramp_limit = u["ramp_rate_limit_pct_min"]["value"]  # %/min
        self.PLR_min = u["PLR_min"]["value"]
        self.PLR_max = u["PLR_max"]["value"]
        self.total_margin_pcm = u["total_reactivity_margin_pcm"]["value"]
        self.xe_react_coeff = u["xenon_reactivity_coeff_pcm"]["value"]

    def equilibrium_xenon(self, power_fraction):
        """
        Equilibrium Xe-135 concentration (normalized) at given power level.
        At equilibrium: Xe_eq = (gamma_I + gamma_Xe) * Sigma_f * phi / (lambda_Xe + sigma_Xe * phi)
        Normalized so Xe_eq(PLR=1) = 1.0.
        """
        phi = np.asarray(power_fraction, dtype=float)
        phi = np.clip(phi, 0.0, 1.0)

        # Proportional constants (using phi as proxy for neutron flux)
        gamma_total = self.gamma_I + self.gamma_Xe
        Xe_eq = gamma_total * phi / (self.lambda_Xe + self.sigma_Xe * phi)

        # Normalize to full-power equilibrium
        Xe_eq_full = gamma_total * 1.0 / (self.lambda_Xe + self.sigma_Xe * 1.0)
        if Xe_eq_full > 0:
            Xe_eq = Xe_eq / Xe_eq_full

        return Xe_eq

    def xenon_transient(self, previous_power_fraction, new_power_fraction,
                        time_hours):
        """
        Compute xenon concentration after a power change.

        After reducing power from P1 to P2:
          - I-135 decays toward new (lower) equilibrium
          - Xe-135 initially builds up because I->Xe production exceeds
            Xe destruction (less neutron flux to burn Xe)
          - Peak occurs at ~8-12 hours

        Simplified analytical model for the xenon transient using
        the linearized solution.

        Returns:
            Xe_relative: Xe concentration relative to full-power equilibrium
        """
        P1 = np.clip(float(previous_power_fraction), 0.0, 1.0)
        P2 = np.clip(float(new_power_fraction), 0.0, 1.0)
        t = float(time_hours) * 3600.0  # convert to seconds

        # Equilibrium values at P1 and P2 (normalized)
        Xe_eq_1 = float(self.equilibrium_xenon(P1))
        Xe_eq_2 = float(self.equilibrium_xenon(P2))
        I_eq_1 = float(P1)  # I-135 equilibrium proportional to power

        # Effective Xe destruction rate at new power level
        lambda_eff = self.lambda_Xe + self.sigma_Xe * P2

        # I-135 transient: I(t) = I_eq_2 + (I_eq_1 - I_eq_2) * exp(-lambda_I * t)
        I_t = P2 + (P1 - P2) * np.exp(-self.lambda_I * t)

        # Xe-135 transient (simplified two-exponential model):
        # Xe(t) = Xe_eq_2 + A1*exp(-lambda_I*t) + A2*exp(-lambda_eff*t)
        # The key physics: after power reduction, Xe peaks then decays

        dI = P1 - P2  # Change in iodine source
        dXe = Xe_eq_1 - Xe_eq_2

        if abs(lambda_eff - self.lambda_I) > 1e-15:
            A1 = self.lambda_I * dI / (lambda_eff - self.lambda_I)
            A2 = dXe - A1
        else:
            A1 = 0.0
            A2 = dXe

        Xe_t = Xe_eq_2 + A1 * np.exp(-self.lambda_I * t) + A2 * np.exp(-lambda_eff * t)

        return max(0.0, float(Xe_t))

    def xenon_reactivity_pcm(self, xe_relative):
        """
        Reactivity penalty from xenon [pcm].
        At full-power equilibrium (xe_relative=1.0), penalty = xe_react_coeff (-3000 pcm).
        """
        return self.xe_react_coeff * xe_relative

    def available_reactivity_pcm(self, xe_relative):
        """
        Available reactivity = total margin + xenon reactivity (which is negative).
        If available < 0, reactor cannot maintain criticality.
        """
        return self.total_margin_pcm + self.xenon_reactivity_pcm(xe_relative)

    def can_restart(self, xe_relative):
        """
        Whether the reactor can return to full power given current Xe level.
        Requires available reactivity > 0.
        """
        return self.available_reactivity_pcm(xe_relative) > 0

    def ramp_rate_limit(self, current_power, target_power, time_minutes):
        """
        Check if a power change is within ramp rate limits.

        Returns:
            achievable_power: the power fraction achievable within the time
            limited: whether the ramp rate was limiting
        """
        current = float(current_power)
        target = float(target_power)
        dt_min = float(time_minutes)

        max_change = self.ramp_limit / 100.0 * dt_min  # fraction change allowed

        if abs(target - current) <= max_change:
            return target, False
        else:
            if target > current:
                return current + max_change, True
            else:
                return current - max_change, True

    def predict(self, power_fraction, time_at_power_hours,
                previous_power_fraction):
        """
        Compute PWR load-following state.

        Args:
            power_fraction:          Current power level [0.3-1.0]
            time_at_power_hours:     Time since last power change [hours]
            previous_power_fraction: Power level before the change [0-1.0]

        Returns:
            dict with:
                power_output_mw      : Electrical output [MW_e]
                xenon_concentration_rel : Xe relative to full-power equilibrium
                available_reactivity_pcm : Available reactivity [pcm]
                ramp_rate_limit_pct_min : Maximum ramp rate [%/min]
                can_restart          : Whether return to full power is possible
        """
        PLR = np.clip(float(power_fraction), self.PLR_min, self.PLR_max)
        t_h = float(time_at_power_hours)
        P_prev = np.clip(float(previous_power_fraction), 0.0, 1.0)

        # Electrical output
        P_electric = self.P_thermal * PLR * self.eta

        # Xenon state
        Xe_rel = self.xenon_transient(P_prev, PLR, t_h)

        # Available reactivity
        avail_pcm = self.available_reactivity_pcm(Xe_rel)

        # Can restart?
        restart_ok = self.can_restart(Xe_rel)

        return {
            "power_output_mw": float(P_electric),
            "xenon_concentration_rel": float(Xe_rel),
            "available_reactivity_pcm": float(avail_pcm),
            "ramp_rate_limit_pct_min": float(self.ramp_limit),
            "can_restart": bool(restart_ok),
        }
