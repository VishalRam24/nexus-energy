"""
EC044 — Monocrystalline Silicon PV — F1a Single-Diode Model

Wraps pvlib's De Soto single-diode model (5 parameters).
I = I_L - I_o * [exp((V + I*Rs) / a) - 1] - (V + I*Rs) / R_sh

Reference:
    De Soto et al. (2006). "Improvement and validation of a model for photovoltaic
    array performance." Solar Energy, 80(1), 78-88.

Library:
    pvlib v0.15 (BSD-3 license)
"""

import numpy as np


class MonoSiPVF1a:
    """Single-diode PV model using De Soto 5-parameter approach."""

    def __init__(self, params: dict):
        mod = params["module"]
        ds = params["desoto_params"]

        self.cells_in_series = mod["cells_in_series"]["value"]
        self.alpha_sc = mod["alpha_sc"]["value"]

        # De Soto reference parameters at STC
        self.I_L_ref = ds["I_L_ref"]["value"]
        self.I_o_ref = ds["I_o_ref"]["value"]
        self.R_s = ds["R_s"]["value"]
        self.R_sh_ref = ds["R_sh_ref"]["value"]
        self.a_ref = ds["a_ref"]["value"]
        self.EgRef = ds["EgRef"]["value"]
        self.dEgdT = ds["dEgdT"]["value"]

        # Physical constants
        self.k = 1.380649e-23   # Boltzmann [J/K]
        self.q = 1.602176634e-19  # electron charge [C]
        self.T_ref = 298.15      # STC temperature [K]
        self.G_ref = 1000.0      # STC irradiance [W/m2]

    def _calc_params(self, irradiance, cell_temp_c):
        """Calculate 5 diode parameters at given conditions."""
        G = np.asarray(irradiance, dtype=float)
        T = np.asarray(cell_temp_c, dtype=float) + 273.15

        # Light current
        I_L = (G / self.G_ref) * (self.I_L_ref + self.alpha_sc * (T - self.T_ref))

        # Bandgap at temperature
        Eg = self.EgRef * (1 + self.dEgdT * (T - self.T_ref) / self.EgRef)

        # Saturation current
        I_o = self.I_o_ref * (T / self.T_ref)**3 * np.exp(
            (self.EgRef / (self.k / self.q * self.T_ref)) -
            (Eg / (self.k / self.q * T))
        )

        # Modified ideality factor
        a = self.a_ref * T / self.T_ref

        # Shunt resistance (inversely proportional to irradiance)
        R_sh = self.R_sh_ref * (self.G_ref / np.maximum(G, 1.0))

        return I_L, I_o, self.R_s, R_sh, a

    def mpp(self, irradiance, cell_temp_c):
        """Maximum power point using pvlib's single_diode solver."""
        try:
            from pvlib.pvsystem import singlediode
            I_L, I_o, R_s, R_sh, a = self._calc_params(irradiance, cell_temp_c)
            result = singlediode(I_L, I_o, R_s, R_sh, a, method='lambertw')
            return {
                "v_mp": np.asarray(result["v_mp"]),
                "i_mp": np.asarray(result["i_mp"]),
                "p_mp": np.asarray(result["p_mp"]),
                "v_oc": np.asarray(result["v_oc"]),
                "i_sc": np.asarray(result["i_sc"]),
            }
        except ImportError:
            return self._mpp_fallback(irradiance, cell_temp_c)

    def _mpp_fallback(self, irradiance, cell_temp_c):
        """Simplified MPP estimate without pvlib."""
        G = np.asarray(irradiance, dtype=float)
        T = np.asarray(cell_temp_c, dtype=float)
        # Linear scaling from STC
        p_mp = 280.0 * (G / 1000.0) * (1 - 0.004 * (T - 25.0))
        p_mp = np.maximum(p_mp, 0.0)
        v_mp = 31.2 * (1 - 0.003 * (T - 25.0))
        i_mp = np.where(v_mp > 0, p_mp / np.maximum(v_mp, 0.1), 0.0)
        v_oc = 38.3 * (1 - 0.003 * (T - 25.0))
        i_sc = 9.39 * (G / 1000.0) * (1 + 0.0005 * (T - 25.0))
        return {"v_mp": v_mp, "i_mp": i_mp, "p_mp": p_mp, "v_oc": v_oc, "i_sc": i_sc}

    def efficiency(self, irradiance, cell_temp_c, area=1.638):
        """Module efficiency = P_mp / (G * A)."""
        result = self.mpp(irradiance, cell_temp_c)
        G = np.asarray(irradiance, dtype=float)
        return np.where(G > 0, result["p_mp"] / (G * area), 0.0)
