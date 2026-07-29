"""
EC071 — Absorption Heat Pump — F1a COP Map Model

Single-effect, thermally driven Type-I absorption heat pump (LiBr-H2O).
The reversible (ideal) heating COP is:

    COP_rev_h = (T_gen - T_cond) / T_gen * T_cond / (T_cond - T_evap) + 1

i.e. the cascade of a Carnot engine driven between T_gen and T_cond
producing work that drives a Carnot heat pump between T_evap and T_cond.
The actual COP is a fraction of this reversible limit:

    COP_h = eta_rev * COP_rev_h

with eta_rev (the "fraction of reversible") around 0.5–0.6 for real
single-effect units. Heating output:

    Q_heating = Q_rated
    Q_gen     = Q_heating / COP_h        (driving heat input)
    Q_evap    = Q_heating - Q_gen        (low-grade source extraction)

Reference:
    Herold, Radermacher & Klein (2016). Absorption Chillers and Heat Pumps,
    2nd ed., CRC Press.
    Hellmann, H.-M., Ziegler, F. (1999). Int. J. Refrigeration 22, 552-560.
"""

import numpy as np


class AbsorptionHeatPumpF1a:
    """Single-effect Type-I absorption heat pump — characteristic-eq style COP."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.rated_capacity = u["rated_heating_capacity"]["value"]   # kW_th
        self.eta_rev        = u["carnot_fraction"]["value"]          # fraction of COP_rev
        self.aux_power      = u["auxiliary_power"]["value"]          # kW_e

    def cop_reversible_heating(self, T_gen_c, T_evap_c, T_cond_c):
        """Reversible (cascade-Carnot) heating COP of an absorption HP."""
        T_gen  = np.asarray(T_gen_c,  dtype=float) + 273.15
        T_evap = np.asarray(T_evap_c, dtype=float) + 273.15
        T_cond = np.asarray(T_cond_c, dtype=float) + 273.15
        dT_lift = T_cond - T_evap
        dT_drive = T_gen - T_cond
        # Guard divisions
        dT_lift_safe  = np.where(dT_lift  > 1e-6, dT_lift,  1e-6)
        T_gen_safe    = np.where(T_gen    > 1e-6, T_gen,    1e-6)
        # Heating COP (Type-I): eta_carnot_engine * COP_carnot_HP + 1
        eta_engine = np.maximum(dT_drive / T_gen_safe, 0.0)
        cop_carnot_hp = T_cond / dT_lift_safe
        return eta_engine * cop_carnot_hp + 1.0

    def cop(self, T_gen_c, T_evap_c, T_cond_c):
        """Actual heating COP = eta_rev * COP_reversible."""
        cop_rev = self.cop_reversible_heating(T_gen_c, T_evap_c, T_cond_c)
        cop = self.eta_rev * cop_rev
        return np.clip(cop, 0.3, 2.5)

    def heating_capacity(self, T_gen_c, T_evap_c, T_cond_c, plr=1.0):
        # Broadcast against the temperature inputs so callers always get an
        # array of the correct shape (matches the COP / energy flows).
        broadcast = (np.asarray(T_gen_c, dtype=float)
                     + np.asarray(T_evap_c, dtype=float)
                     + np.asarray(T_cond_c, dtype=float)) * 0.0
        return self.rated_capacity * np.asarray(plr, dtype=float) + broadcast

    def driving_heat(self, T_gen_c, T_evap_c, T_cond_c, plr=1.0):
        """Generator (driving) heat input in kW_th."""
        q = self.heating_capacity(T_gen_c, T_evap_c, T_cond_c, plr)
        c = self.cop(T_gen_c, T_evap_c, T_cond_c)
        return q / c

    def evaporator_heat(self, T_gen_c, T_evap_c, T_cond_c, plr=1.0):
        """Heat extracted at the evaporator (low-grade source) in kW_th."""
        q_h   = self.heating_capacity(T_gen_c, T_evap_c, T_cond_c, plr)
        q_gen = self.driving_heat(T_gen_c, T_evap_c, T_cond_c, plr)
        return np.maximum(q_h - q_gen, 0.0)

    def electrical_input(self, plr=1.0):
        """Auxiliary electrical input (pumps, controls) in kW_e."""
        plr = np.asarray(plr, dtype=float)
        return self.aux_power * np.where(plr > 0, 1.0, 0.0) + 0.0 * plr
