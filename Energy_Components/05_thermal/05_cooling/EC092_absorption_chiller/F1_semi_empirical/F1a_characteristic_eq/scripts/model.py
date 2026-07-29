"""
EC092 — Absorption Chiller — F1a Characteristic Equation Model

COP = COP_max * (1 - exp(-alpha * dT_driving / dT_ref))
dT_driving = T_generator - T_condenser

Energy balance:
  Q_generator (heat input) drives the cycle.
  Q_cool (cooling delivered to chilled water) = COP * Q_generator
  Q_reject (heat to cooling tower) = Q_generator + Q_cool  [first law]

References:
    Herold, Radermacher & Klein (2016), 'Absorption Chillers and Heat Pumps',
    2nd ed., CRC Press.
    Gordon & Ng (2000), 'Cool Thermodynamics', Cambridge International Science.
"""

import numpy as np


class AbsorptionChillerF1a:
    """Single-effect LiBr-H2O absorption chiller — characteristic equation."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_cool_rated = u["Q_cool_rated"]["value"]   # kW
        self.COP_max = u["COP_max"]["value"]             # dimensionless
        self.alpha = u["alpha"]["value"]                 # shape factor
        self.dT_ref = u["dT_ref"]["value"]              # K

    def cop(self, T_gen_c, T_cond_c):
        """COP as function of generator and condenser temperatures."""
        T_gen = np.asarray(T_gen_c, dtype=float)
        T_cond = np.asarray(T_cond_c, dtype=float)
        dT_driving = T_gen - T_cond
        cop_val = self.COP_max * (1.0 - np.exp(-self.alpha * dT_driving / self.dT_ref))
        # Physical bounds: COP must be >= 0; single-effect LiBr-H2O cap at 0.80
        return np.clip(cop_val, 0.0, 0.80)

    def heat_flows(self, T_gen_c, T_cond_c, Q_cool_kw=None):
        """
        Compute all heat flows given operating temperatures.

        Parameters
        ----------
        T_gen_c   : generator temperature (degC)
        T_cond_c  : condenser temperature (degC)
        Q_cool_kw : override cooling load (kW); defaults to rated capacity

        Returns
        -------
        dict with cop, Q_generator_kw, Q_cool_kw, Q_reject_kw
        """
        cop_val = self.cop(T_gen_c, T_cond_c)
        if Q_cool_kw is None:
            Q_cool = np.full_like(cop_val, self.Q_cool_rated)
        else:
            Q_cool = np.asarray(Q_cool_kw, dtype=float) * np.ones_like(cop_val)

        # Avoid divide-by-zero at very low COP (startup / off-design)
        safe_cop = np.where(cop_val > 0.01, cop_val, 0.01)
        Q_gen = Q_cool / safe_cop
        Q_reject = Q_gen + Q_cool   # First law: Q_reject = Q_gen + Q_evap

        return {
            "cop": cop_val,
            "Q_generator_kw": Q_gen,
            "Q_cool_kw": Q_cool,
            "Q_reject_kw": Q_reject,
        }
