"""
EC071 — Absorption Heat Pump — F1b Part-Load Model

Extends F1a (cascade-Carnot COP) with:

  1. Generator temperature sensitivity:
       f_T_gen = exp(-beta * max(T_gen_design - T_gen, 0))
     COP falls when T_gen drops below design because the desorption driving
     force (concentration difference) decreases, reducing refrigerant lift.
     Reference: Hellmann & Ziegler (1999), Int. J. Refrigeration 22, 552-560.

  2. Part-load degradation of driving-heat utilization (EN 14825 analogy):
       PLF = 1 - C_d * (1 - PLR)
     COP_pl = COP_full * PLF
     At PLR=1: PLF=1.  At part load the generator is overfilled relative to
     the required desorption duty, reducing effective utilization.
     Reference: Jakob et al. (2008), IEA Annex 34 AHP simulation benchmarks.

  3. On/off cycling penalty below PLR_min:
     Additional transient losses from generator/absorber thermal mass.

References:
    Hellmann & Ziegler (1999). Int. J. Refrigeration 22, 552-560.
    Herold, Radermacher & Klein (2016). Absorption Chillers and Heat Pumps, 2nd ed., CRC Press.
    Jakob et al. (2008). IEA/ECBCS Annex 34 — Absorption heat pump simulation.
"""

import numpy as np


class AbsorptionHeatPumpF1b:
    """Single-effect LiBr-H2O absorption HP with part-load & T_gen effects."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.rated_capacity = u["rated_heating_capacity"]["value"]   # kW_th
        self.eta_rev        = u["carnot_fraction"]["value"]
        self.aux_power      = u["auxiliary_power"]["value"]          # kW_e
        self.T_gen_design   = u["T_gen_design"]["value"]             # degC
        self.C_d            = u["C_d"]["value"]
        self.PLR_min        = u["PLR_min"]["value"]
        self.cycling_loss   = u["cycling_loss_factor"]["value"]
        self.beta_T_gen     = u["beta_T_gen"]["value"]               # 1/K

    # ------------------------------------------------------------------
    # Reversible COP (same as F1a)
    # ------------------------------------------------------------------

    def _cop_reversible(self, T_gen_c, T_evap_c, T_cond_c):
        T_gen  = np.asarray(T_gen_c,  dtype=float) + 273.15
        T_evap = np.asarray(T_evap_c, dtype=float) + 273.15
        T_cond = np.asarray(T_cond_c, dtype=float) + 273.15
        dT_lift  = np.where(T_cond - T_evap > 1e-6, T_cond - T_evap, 1e-6)
        dT_drive = T_gen - T_cond
        eta_engine = np.maximum(dT_drive / np.where(T_gen > 1e-6, T_gen, 1e-6), 0.0)
        cop_carnot_hp = T_cond / dT_lift
        return eta_engine * cop_carnot_hp + 1.0

    # ------------------------------------------------------------------
    # Generator temperature correction factor
    # ------------------------------------------------------------------

    def f_T_gen(self, T_gen_c):
        """
        COP penalty when T_gen < T_gen_design.

        f_T_gen = exp(-beta * max(T_gen_design - T_gen, 0))

        At design T_gen: f=1.  As T_gen drops, desorption weakens
        and COP falls exponentially.
        """
        T_gen = np.asarray(T_gen_c, dtype=float)
        deficit = np.maximum(self.T_gen_design - T_gen, 0.0)
        return np.exp(-self.beta_T_gen * deficit)

    # ------------------------------------------------------------------
    # Part-load factor
    # ------------------------------------------------------------------

    def part_load_factor(self, plr):
        """PLF = 1 - C_d * (1 - PLR).  At PLR=1: PLF=1."""
        plr = np.asarray(plr, dtype=float)
        plf = 1.0 - self.C_d * (1.0 - plr)
        return np.clip(plf, 0.1, 1.0)

    # ------------------------------------------------------------------
    # Full COP with all corrections
    # ------------------------------------------------------------------

    def cop(self, T_gen_c, T_evap_c, T_cond_c, plr=1.0):
        """
        Actual heating COP including T_gen sensitivity and part-load:

            COP = eta_rev * COP_rev * f_T_gen * PLF * f_cycling
        """
        plr  = np.asarray(plr, dtype=float)
        cop_rev = self._cop_reversible(T_gen_c, T_evap_c, T_cond_c)
        f_gen = self.f_T_gen(T_gen_c)
        plf   = self.part_load_factor(plr)

        cop_full = self.eta_rev * cop_rev * f_gen

        # Cycling penalty below PLR_min
        cycling_penalty = np.where(
            plr < self.PLR_min,
            1.0 - self.cycling_loss * (self.PLR_min - plr) / self.PLR_min,
            1.0,
        )

        cop_pl = cop_full * plf * cycling_penalty
        return np.clip(cop_pl, 0.3, 2.5)

    def cop_degradation_factor(self, T_gen_c, plr):
        """Combined degradation: f_T_gen * PLF * f_cycling relative to design full load."""
        plr = np.asarray(plr, dtype=float)
        f_gen = self.f_T_gen(T_gen_c)
        plf   = self.part_load_factor(plr)
        cycling_penalty = np.where(
            plr < self.PLR_min,
            1.0 - self.cycling_loss * (self.PLR_min - plr) / self.PLR_min,
            1.0,
        )
        return f_gen * plf * cycling_penalty

    # ------------------------------------------------------------------
    # Capacity and energy flows
    # ------------------------------------------------------------------

    def heating_capacity(self, plr=1.0):
        """Heating output at given PLR [kW_th]."""
        return self.rated_capacity * np.asarray(plr, dtype=float)

    def driving_heat(self, T_gen_c, T_evap_c, T_cond_c, plr=1.0):
        """Generator (driving) heat input [kW_th]."""
        q = self.heating_capacity(plr)
        c = self.cop(T_gen_c, T_evap_c, T_cond_c, plr)
        return q / np.where(c > 1e-6, c, 1e-6)

    def evaporator_heat(self, T_gen_c, T_evap_c, T_cond_c, plr=1.0):
        """Low-grade heat extracted at evaporator [kW_th]."""
        q_h   = self.heating_capacity(plr)
        q_gen = self.driving_heat(T_gen_c, T_evap_c, T_cond_c, plr)
        return np.maximum(q_h - q_gen, 0.0)

    def electrical_input(self, plr=1.0):
        """Auxiliary electrical input [kW_e]."""
        plr = np.asarray(plr, dtype=float)
        return self.aux_power * np.where(plr > 0, 1.0, 0.0) + 0.0 * plr
