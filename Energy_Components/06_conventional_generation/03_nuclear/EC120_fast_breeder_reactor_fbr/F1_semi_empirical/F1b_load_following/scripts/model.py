"""
EC120 -- Fast Breeder Reactor (FBR) -- F1b Load-Following Model

Extends FBR power map with:
  1. Xe-135 / I-135 dynamics -- NEGLIGIBLE in fast spectrum
     Fast neutron cross-section of Xe-135 is ~10,000x smaller than thermal.
     Xe reactivity worth in FBR is ~50 pcm vs ~3000 pcm in PWR.
     Xenon deadtime does NOT affect FBR load-following.
     (Model still tracks Xe for completeness; it has no practical impact.)

  2. Sodium void reactivity feedback (positive in large FBR cores):
     rho_void = sodium_void_coeff * delta_void_pct
     where delta_void ~ alpha_void * (1 - PLR)
     This positive feedback is a key FBR safety concern but is controlled
     by design margins and negative Doppler coefficient.

  3. Doppler temperature feedback (negative, stabilizing):
     rho_Doppler = doppler_coeff * delta_T_Na
     where delta_T_Na = T_outlet_rated * (1 - PLR)
     Doppler effect in U-238 rich fuel provides passive power self-limitation.

  4. Combined sodium feedback and Doppler determines overall power coefficient;
     well-designed FBR has net negative power coefficient at operating points.

  5. Ramp rate limited by:
     - Thermal gradients in large sodium pool (thermal stratification risk)
     - Breeding blanket structural integrity
     - ~3%/min operational limit (BN-800 experience)

Key FBR vs PWR differences:
  - Xe poisoning negligible (fast spectrum σ_Xe ~10000x smaller)
  - Positive sodium void coefficient (compensated by negative Doppler)
  - Slower ramp rate (3%/min vs 5%/min PWR) due to large sodium pool
  - High breeding ratio enables extended fuel cycles

References:
    Guidez, J. & Prele, G. (2017). Sodium Cooled Fast Reactors.
        EDP Sciences, Les Ulis, France.
    Koch, L.J. (2008). Experimental Breeder Reactor-II. ANL Publication.
    IAEA (2012). Status of Fast Reactor Technology.
        IAEA-TECDOC-1689. Vienna.
    Stacey, W.M. (2007). Nuclear Reactor Physics, 2nd ed. Wiley.
"""

import numpy as np


class FBRF1b:
    """FBR (sodium-cooled) load-following model with negligible Xe and sodium void feedback."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_thermal       = u["P_thermal_mw"]["value"]
        self.eta             = u["eta_thermal"]["value"]
        self.sigma_Xe        = u["sigma_Xe_cm2"]["value"]     # fast spectrum: very small
        self.lambda_Xe       = u["lambda_Xe_per_s"]["value"]
        self.lambda_I        = u["lambda_I_per_s"]["value"]
        self.gamma_I         = u["gamma_I"]["value"]
        self.gamma_Xe        = u["gamma_Xe"]["value"]
        self.ramp_limit      = u["ramp_rate_limit_pct_min"]["value"]
        self.PLR_min         = u["PLR_min"]["value"]
        self.PLR_max         = u["PLR_max"]["value"]
        self.total_margin    = u["total_reactivity_margin_pcm"]["value"]
        self.xe_react_coeff  = u["xenon_reactivity_coeff_pcm"]["value"]   # ~-50 pcm (negligible)
        self.void_coeff      = u["sodium_void_coeff_pcm_per_pct"]["value"]  # +10 pcm/% (positive!)
        self.doppler_coeff   = u["doppler_coeff_pcm_per_K"]["value"]        # -0.8 pcm/K
        self.T_sodium_rated  = u["T_sodium_rated_C"]["value"]

        # Void change per unit PLR change (linear approximation)
        # At full power: core void fraction ~2% (small bubbles in sodium)
        # At part load: void decreases as power/temperature drops
        self._void_per_plr = 2.0  # % void per unit PLR (linear)

    # ------------------------------------------------------------------
    # Xenon dynamics (negligible in fast spectrum)
    # ------------------------------------------------------------------

    def equilibrium_xenon(self, power_fraction):
        """
        Normalized equilibrium Xe-135 (fast spectrum).
        Due to tiny fast-spectrum σ_Xe, the burnup term dominates only through
        decay. Xe_eq is much smaller relative to thermal reactor case.
        Normalized to 1.0 at full power (reference value for tracking).
        """
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
        Xe-135 transient (negligible reactivity impact in fast spectrum).
        Uses same analytical model as PWR F1b, but sigma_Xe is ~10000x smaller
        so the effective Xe reactivity worth is ~50 pcm (negligible).
        Tracked for completeness / safety record.
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
    # Sodium void feedback (FBR-specific)
    # ------------------------------------------------------------------

    def void_change_pct(self, current_plr, reference_plr=1.0):
        """
        Change in average core void fraction relative to rated.
        At part load, sodium temperature drops, void fraction decreases.
        Returns delta_void in % (negative = less void = positive reactivity).
        """
        return self._void_per_plr * (float(current_plr) - float(reference_plr))

    def sodium_void_reactivity_pcm(self, current_plr, reference_plr=1.0):
        """
        Sodium void reactivity feedback [pcm].
        Positive void coefficient: less void (part load) → positive reactivity.
        rho_void = void_coeff [pcm/%void] * delta_void [%void]
        NOTE: delta_void is negative at part load → rho_void is negative here
        (less void → less positive reactivity → slight net negative contribution
        at part load). This is a conservative treatment.
        """
        delta_void = self.void_change_pct(current_plr, reference_plr)
        return self.void_coeff * delta_void

    # ------------------------------------------------------------------
    # Doppler feedback
    # ------------------------------------------------------------------

    def sodium_temp_change_K(self, power_fraction):
        """
        Approximate change in average sodium outlet temperature.
        T_out ~ T_rated * PLR (linear approximation).
        """
        plr = np.clip(np.asarray(power_fraction, dtype=float), 0.0, 1.0)
        return self.T_sodium_rated * (float(plr) - 1.0)  # negative at part load

    def doppler_reactivity_pcm(self, power_fraction):
        """
        Doppler temperature reactivity [pcm].
        rho_Doppler = doppler_coeff * delta_T
        At part load (T drops), negative Doppler adds positive reactivity,
        partially offsetting the slightly positive sodium void change.
        """
        delta_T = self.sodium_temp_change_K(power_fraction)
        return float(self.doppler_coeff) * float(delta_T)

    # ------------------------------------------------------------------
    # Reactivity balance
    # ------------------------------------------------------------------

    def xenon_reactivity_pcm(self, xe_relative):
        """Xe reactivity [pcm] -- negligible in fast spectrum but tracked."""
        return self.xe_react_coeff * xe_relative

    def available_reactivity_pcm(self, xe_relative, power_fraction):
        """
        Available reactivity = total margin + Xe penalty (tiny) + void + Doppler.
        For FBR, Xe term is negligible; the void-Doppler balance dominates.
        """
        rho_Xe      = self.xenon_reactivity_pcm(xe_relative)
        rho_void    = self.sodium_void_reactivity_pcm(power_fraction)
        rho_Doppler = self.doppler_reactivity_pcm(power_fraction)
        return self.total_margin + rho_Xe + rho_void + rho_Doppler

    def can_restart(self, xe_relative, power_fraction=1.0):
        """Whether available reactivity > 0 for target power level."""
        return self.available_reactivity_pcm(xe_relative, power_fraction) > 0

    # ------------------------------------------------------------------
    # Ramp rate
    # ------------------------------------------------------------------

    def ramp_rate_limit(self, current_power, target_power, time_minutes):
        """Achievable power change within FBR ramp rate limit (3%/min)."""
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
        Compute FBR load-following state.

        Returns dict with:
            power_output_mw           : Electrical output [MW_e]
            xenon_concentration_rel   : Xe relative to full-power eq (negligible impact)
            sodium_void_reactivity_pcm: Sodium void feedback [pcm]
            doppler_reactivity_pcm    : Doppler feedback [pcm]
            available_reactivity_pcm  : Net available reactivity [pcm]
            ramp_rate_limit_pct_min   : Maximum ramp rate [%/min]
            can_restart               : Whether restart is possible
        """
        PLR    = np.clip(float(power_fraction), self.PLR_min, self.PLR_max)
        t_h    = float(time_at_power_hours)
        P_prev = np.clip(float(previous_power_fraction), 0.0, 1.0)

        P_electric   = self.P_thermal * PLR * self.eta
        Xe_rel       = self.xenon_transient(P_prev, PLR, t_h)
        rho_void     = self.sodium_void_reactivity_pcm(PLR)
        rho_Doppler  = self.doppler_reactivity_pcm(PLR)
        avail_pcm    = self.available_reactivity_pcm(Xe_rel, PLR)
        restart_ok   = self.can_restart(Xe_rel, PLR)

        return {
            "power_output_mw":            float(P_electric),
            "xenon_concentration_rel":    float(Xe_rel),
            "sodium_void_reactivity_pcm": float(rho_void),
            "doppler_reactivity_pcm":     float(rho_Doppler),
            "available_reactivity_pcm":   float(avail_pcm),
            "ramp_rate_limit_pct_min":    float(self.ramp_limit),
            "can_restart":                bool(restart_ok),
        }
