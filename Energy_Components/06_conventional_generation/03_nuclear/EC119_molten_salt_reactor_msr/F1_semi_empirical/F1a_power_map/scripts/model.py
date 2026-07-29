"""
EC119 -- Molten Salt Reactor (MSR) -- F1a Power-Map Model

Simplest semi-empirical model:
    P_elec = eta * P_thermal * load_factor

No xenon dynamics, no temperature feedback, no ramp-rate constraints.
F1b adds those phenomena.

Key MSR characteristics captured here:
  - High thermal efficiency (~45%) enabled by high core temperatures (700 degC)
  - Linear scaling of electrical output with load factor
  - Load factor bounded by design limits [0.3, 1.0]

References:
    Haubenreich, P.N. & Engel, J.R. (1970). Nucl. Appl. Technol. 8(2):118.
    MacPherson, H.G. (1985). Nucl. Sci. Eng. 90:374.
    NEA (2021). Molten Salt Reactor Technology Handbook.
"""

import numpy as np


class MSRF1a:
    """MSR power-map: P_elec = eta * P_thermal * load_factor."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_thermal = float(u["P_thermal_mw"]["value"])   # MW_th
        self.eta       = float(u["eta_thermal"]["value"])
        self.T_core    = float(u["T_core_degC"]["value"])
        self.load_min  = float(u["load_min"]["value"])
        self.load_max  = float(u["load_max"]["value"])

    def predict(self, load_factor: float) -> dict:
        """
        Parameters
        ----------
        load_factor : float
            Fraction of rated thermal power [0.3 – 1.0]

        Returns
        -------
        dict with:
            load_factor_clamped  : effective load factor after clamping
            P_thermal_mw         : thermal power output [MW_th]
            P_electric_mw        : electrical output [MW_e]
            eta_thermal          : thermal-to-electric efficiency [-]
        """
        lf = float(np.clip(load_factor, self.load_min, self.load_max))
        P_th  = self.P_thermal * lf
        P_el  = self.eta * P_th

        return {
            "load_factor_clamped": float(lf),
            "P_thermal_mw":        float(P_th),
            "P_electric_mw":       float(P_el),
            "eta_thermal":         float(self.eta),
        }
