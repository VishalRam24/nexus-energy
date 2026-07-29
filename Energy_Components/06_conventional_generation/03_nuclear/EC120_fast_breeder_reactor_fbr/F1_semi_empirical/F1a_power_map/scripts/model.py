"""
EC120 -- Fast Breeder Reactor (FBR) -- F1a Power-Map Model

Simplest semi-empirical model:
    P_elec = eta * P_thermal * load_factor

Sodium-cooled fast reactor. No xenon dynamics (Xe cross-section negligible
in fast spectrum), no sodium void feedback. F1b adds those phenomena.

Key FBR characteristics:
  - Sodium-cooled, fast-neutron spectrum
  - Core outlet temperature ~550 degC
  - Net efficiency ~40% (steam Rankine at 500 degC)
  - Xenon poisoning negligible (fast spectrum: sigma_Xe ~3 orders lower)

References:
    Guidez, J. & Prele, G. (2017). Sodium Cooled Fast Reactors. EDP Sciences.
    IAEA (2012). Status of Fast Reactor Technology. IAEA-TECDOC-1689.
    Koch, L.J. (2008). Fast Breeder Reactors. Elsevier.
"""

import numpy as np


class FBRF1a:
    """FBR power-map: P_elec = eta * P_thermal * load_factor."""

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
            Fraction of rated thermal power [0.25 – 1.0]

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
