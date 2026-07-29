"""
EC121 -- High Temperature Gas Reactor (HTGR) -- F1a Power-Map Model

Simplest semi-empirical model:
    P_elec = eta * P_thermal * load_factor

Helium-cooled reactor. No xenon dynamics, no graphite thermal-mass effects.
F1b adds those phenomena.

Key HTGR characteristics:
  - Helium coolant, TRISO fuel, graphite moderator
  - Core outlet temperature ~750 degC
  - Very high net efficiency ~47% (Brayton/Rankine at high temperature)
  - Can supply industrial process heat as well as electricity

References:
    Zhang, Z. et al. (2009). Nucl. Eng. Des. 239:2265.
    Dong, Y. (2011). Nucl. Eng. Des. 241:4755.
    Reitsma, F. (2004). IAEA TRS-424.
"""

import numpy as np


class HTGRF1a:
    """HTGR power-map: P_elec = eta * P_thermal * load_factor."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_thermal = float(u["P_thermal_mw"]["value"])   # MW_th
        self.eta       = float(u["eta_thermal"]["value"])
        self.T_out     = float(u["T_out_degC"]["value"])
        self.load_min  = float(u["load_min"]["value"])
        self.load_max  = float(u["load_max"]["value"])

    def predict(self, load_factor: float) -> dict:
        """
        Parameters
        ----------
        load_factor : float
            Fraction of rated thermal power [0.4 – 1.0]

        Returns
        -------
        dict with:
            load_factor_clamped  : effective load factor after clamping
            P_thermal_mw         : thermal power output [MW_th]
            P_electric_mw        : electrical output [MW_e]
            eta_thermal          : thermal-to-electric efficiency [-]
        """
        lf   = float(np.clip(load_factor, self.load_min, self.load_max))
        P_th = self.P_thermal * lf
        P_el = self.eta * P_th

        return {
            "load_factor_clamped": float(lf),
            "P_thermal_mw":        float(P_th),
            "P_electric_mw":       float(P_el),
            "eta_thermal":         float(self.eta),
        }
