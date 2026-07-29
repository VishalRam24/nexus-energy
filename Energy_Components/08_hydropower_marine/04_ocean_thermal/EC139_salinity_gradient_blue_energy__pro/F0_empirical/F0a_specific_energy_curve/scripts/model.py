"""EC139 — Salinity Gradient (Blue Energy, PRO) — F0a empirical specific-energy curve.

Simplest fidelity: tabulated extractable specific energy (kWh/m3) and net power (kW)
vs seawater concentration Csw, interpolated with numpy.interp. The underlying relation
is the van't Hoff osmotic pressure scaled by membrane and recovery efficiencies.
Data source: Yip & Elimelech (2012); Straub et al. (2016); reuses EC139 F1a parameters.
NumPy only.
"""
import numpy as np


class SalinityGradientF0a:
    def __init__(self, params):
        r = params["rated"]
        c = params["energy_curve"]
        self.SE_design = r["SE_design_kWh_per_m3"]["value"]
        self.P_design = r["P_net_design_kw"]["value"]
        self.C_tab = np.asarray(c["C_seawater_g_per_L"]["value"], dtype=float)
        self.SE_tab = np.asarray(c["specific_energy_kWh_per_m3"]["value"], dtype=float)
        self.P_tab = np.asarray(c["net_power_kw"]["value"], dtype=float)

    def specific_energy_kwh_m3(self, Csw):
        """Extractable specific energy (kWh/m3) at seawater concentration Csw (g/L)."""
        return np.interp(np.asarray(Csw, dtype=float), self.C_tab, self.SE_tab)

    def net_power_kw(self, Csw):
        return np.interp(np.asarray(Csw, dtype=float), self.C_tab, self.P_tab)
