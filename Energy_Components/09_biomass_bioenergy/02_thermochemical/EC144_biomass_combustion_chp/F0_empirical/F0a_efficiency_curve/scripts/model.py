"""F0a empirical part-load efficiency curve for EC144 biomass combustion CHP.

Black-box model: rated electrical & thermal efficiencies scaled by a part-load
multiplier (np.interp over PLR breakpoints, equal to the F1 polynomial
a0+a1*PLR+a2*PLR^2), with a linear moisture de-rating. NumPy only.

Data source: Obernberger & Thek (2008); EN303-5:2012; Jenkins et al. (1998)
— reused from the EC144 F1b parameter set.
"""
import numpy as np


class EfficiencyCurve:
    def __init__(self, params):
        u = params["rated"]
        self.Q_rated = u["Q_rated_kw"]["value"]
        self.eta_el_ref = u["eta_electrical_ref"]["value"]
        self.eta_th_ref = u["eta_thermal_ref"]["value"]
        self.PLR_min = u["PLR_min"]["value"]
        pc = params["partload_curve"]
        self.PLR_bp = np.asarray(pc["PLR"], float)
        self.mult_bp = np.asarray(pc["multiplier"], float)

    def partload_multiplier(self, PLR):
        PLR = max(self.PLR_min, min(1.0, PLR))
        return float(np.interp(PLR, self.PLR_bp, self.mult_bp))

    def _moisture_factor(self, moisture):
        # wet fuel loses sensible heat to evaporation; ~ -0.8 per moisture fraction
        return max(0.0, 1.0 - 0.8 * moisture)

    def eta_electrical(self, PLR=1.0, moisture=0.0):
        return self.eta_el_ref * self.partload_multiplier(PLR) * self._moisture_factor(moisture)

    def eta_thermal(self, PLR=1.0, moisture=0.0):
        return self.eta_th_ref * self.partload_multiplier(PLR) * self._moisture_factor(moisture)

    def eta_total(self, PLR=1.0, moisture=0.0):
        return self.eta_electrical(PLR, moisture) + self.eta_thermal(PLR, moisture)
