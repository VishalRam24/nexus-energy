"""EC138 — Ocean Thermal Energy Conversion (OTEC) — F0a empirical efficiency curve.

Simplest fidelity: a tabulated net-efficiency curve vs temperature difference dT
(warm minus cold seawater), interpolated with numpy.interp. Net power = eta_net*P_gross.
Below dT_min there is no net generation.
Data source: Vega (2002); Nihous (2007); reuses EC138 F1a parameters. NumPy only.
"""
import numpy as np


class OtecF0a:
    def __init__(self, params):
        r = params["rated"]
        c = params["efficiency_curve"]
        self.P_gross = r["P_gross_kw"]["value"]
        self.dT_min = r["dT_min_c"]["value"]
        self.dT_tab = np.asarray(c["dT_c"]["value"], dtype=float)
        self.eta_tab = np.asarray(c["eta_net"]["value"], dtype=float)

    def net_efficiency(self, dT):
        """Net thermal-to-electric efficiency at temperature difference dT (degC)."""
        dT = np.asarray(dT, dtype=float)
        eta = np.interp(dT, self.dT_tab, self.eta_tab)
        return np.where(dT < self.dT_min, 0.0, eta)

    def net_power_kw(self, dT):
        return self.net_efficiency(dT) * self.P_gross
