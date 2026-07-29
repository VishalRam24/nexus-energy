"""
EC093 — Adsorption Chiller — F1b Part-Load Model

Extends F1a (cascade-Carnot cooling COP) with:

  1. Part-load degradation — incomplete adsorption equilibrium at reduced load.
     Adsorption chillers operate in timed half-cycles; at part load the cycle is
     shortened, silica gel cannot equilibrate, and the uptake per cycle drops.

     Linear PLF term (EN 14825 analogy):
       PLF_linear = 1 - C_d * (1 - PLR)

     Kinetic half-cycle term (exponential saturation):
       tau_pl = tau_design * PLR^n       (shorter half-cycle at part load)
       PLF_kinetic = 1 - exp(-k_eff * tau_pl) / (1 - exp(-k_eff * tau_design))
       k_eff is calibrated so PLF_kinetic = 1.0 at PLR = 1.

     Combined PLF:
       PLF = PLF_linear * PLF_kinetic_normalized

     The kinetic term captures the physics that adsorption chillers have
     disproportionate COP loss at low PLR relative to vapor-compression.

  2. Cycling losses below PLR_min:
     On/off cycling dissipates thermal mass energy in sorbent beds.

References:
    Saha, B.B., Boelman, E.C., Kashiwagi, T. (1995). Heat Recovery Systems & CHP 15, 581-590.
    Wang, R.Z., Oliveira, R.G. (2006). Prog. Energy Combust. Sci. 32, 424-458.
    Duong, X.Q., Cao, N.V., Chung, J.D. (2018). Energy Conv. Mgmt. 158, 77-90.
    Boelman, E.C., Saha, B.B., Kashiwagi, T. (1995). ASHRAE Trans. 101(1), 825-838.
"""

import numpy as np


class AdsorptionChillerF1b:
    """Silica-gel/water adsorption chiller with part-load and cycling losses."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_rated      = u["rated_cooling_capacity"]["value"]
        self.eta_rev      = u["carnot_fraction"]["value"]
        self.aux_power    = u["auxiliary_power"]["value"]
        self.C_d          = u["C_d"]["value"]
        self.PLR_min      = u["PLR_min"]["value"]
        self.cycling_loss = u["cycling_loss_factor"]["value"]
        self.tau_design   = u["half_cycle_time_design"]["value"]
        self.n_exp        = u["tau_penalty_exponent"]["value"]
        # Pre-compute effective rate constant so PLF_kinetic(PLR=1) = 1
        # Using tau_pl = tau_design at PLR=1: PLF_kinetic_at_1 = 1 by construction below
        self._k_eff = 5.0 / self.tau_design  # 5 time-constants to reach ~99% equilibrium

    # ------------------------------------------------------------------
    # Reversible COP (same as F1a)
    # ------------------------------------------------------------------

    def _cop_reversible(self, T_hot_c, T_cool_c, T_chilled_c):
        T_gen  = np.asarray(T_hot_c,     dtype=float) + 273.15
        T_cool = np.asarray(T_cool_c,    dtype=float) + 273.15
        T_chw  = np.asarray(T_chilled_c, dtype=float) + 273.15
        dT_drive = np.where(T_gen - T_cool > 1e-6, T_gen - T_cool, 1e-6)
        dT_lift  = np.where(T_cool - T_chw > 1e-6, T_cool - T_chw, 1e-6)
        eta_engine = dT_drive / T_gen
        cop_carnot = T_chw / dT_lift
        return np.maximum(eta_engine * cop_carnot, 0.0)

    # ------------------------------------------------------------------
    # Part-load factor
    # ------------------------------------------------------------------

    def part_load_factor(self, plr):
        """
        PLF = PLF_linear * PLF_kinetic_norm

        PLF_linear = 1 - C_d*(1-PLR)

        PLF_kinetic_norm: normalized kinetic adsorption completion.
          At PLR=1, tau_pl=tau_design → ratio = 1.
          At low PLR, shorter half-cycle → less uptake → lower COP.

          PLF_kinetic = (1 - exp(-k * tau_pl)) / (1 - exp(-k * tau_design))
          tau_pl = tau_design * PLR^n
        """
        plr = np.asarray(plr, dtype=float)

        # Linear term
        plf_lin = 1.0 - self.C_d * (1.0 - plr)

        # Kinetic term
        tau_pl   = self.tau_design * np.maximum(plr, 1e-6) ** self.n_exp
        denom    = 1.0 - np.exp(-self._k_eff * self.tau_design)
        denom_s  = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
        plf_kin  = (1.0 - np.exp(-self._k_eff * tau_pl)) / denom_s
        plf_kin  = np.clip(plf_kin, 0.0, 1.0)

        plf = plf_lin * plf_kin
        return np.clip(plf, 0.05, 1.0)

    # ------------------------------------------------------------------
    # COP with part-load
    # ------------------------------------------------------------------

    def cop(self, T_hot_c, T_cool_c, T_chilled_c, plr=1.0):
        """
        COP_pl = eta_rev * COP_rev * PLF(PLR) * f_cycling
        """
        plr = np.asarray(plr, dtype=float)
        cop_rev = self._cop_reversible(T_hot_c, T_cool_c, T_chilled_c)
        plf = self.part_load_factor(plr)

        cycling_penalty = np.where(
            plr < self.PLR_min,
            1.0 - self.cycling_loss * (self.PLR_min - plr) / self.PLR_min,
            1.0,
        )

        cop_pl = self.eta_rev * cop_rev * plf * cycling_penalty
        return np.clip(cop_pl, 0.01, 0.85)

    def cop_degradation_factor(self, plr):
        """PLF * f_cycling."""
        plr = np.asarray(plr, dtype=float)
        plf = self.part_load_factor(plr)
        cycling_penalty = np.where(
            plr < self.PLR_min,
            1.0 - self.cycling_loss * (self.PLR_min - plr) / self.PLR_min,
            1.0,
        )
        return plf * cycling_penalty

    # ------------------------------------------------------------------
    # Energy flows
    # ------------------------------------------------------------------

    def cooling_power(self, plr=1.0):
        return self.Q_rated * np.asarray(plr, dtype=float)

    def driving_heat(self, T_hot_c, T_cool_c, T_chilled_c, plr=1.0):
        q = self.cooling_power(plr)
        c = self.cop(T_hot_c, T_cool_c, T_chilled_c, plr)
        return q / np.where(c > 1e-6, c, 1e-6)

    def heat_rejection(self, T_hot_c, T_cool_c, T_chilled_c, plr=1.0):
        q  = self.cooling_power(plr)
        qd = self.driving_heat(T_hot_c, T_cool_c, T_chilled_c, plr)
        return q + qd

    def electrical_input(self, plr=1.0):
        plr = np.asarray(plr, dtype=float)
        return self.aux_power * np.where(plr > 0, 1.0, 0.0) + 0.0 * plr
