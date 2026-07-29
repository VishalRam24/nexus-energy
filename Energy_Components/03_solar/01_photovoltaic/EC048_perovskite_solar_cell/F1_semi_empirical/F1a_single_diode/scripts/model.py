"""
EC048 — Perovskite Solar Cell — F1a Single-Diode Model

Adapts the De Soto 5-parameter single-diode framework (same as EC044 Si)
but with perovskite-specific parameters:
  - Bandgap Eg = 1.55 eV  (vs 1.12 eV for Si)
  - Higher Voc (~1.18 V)  due to wider bandgap
  - Lower Jsc (~25 mA/cm2) due to reduced absorption below 800 nm
  - Ideality factor n = 1.5 (trap-assisted recombination dominant in perovskite)
  - Lab cell area 25 cm2

Uses pvlib.pvsystem.singlediode if available, else analytical fallback.

References:
    De Soto et al. (2006). Solar Energy, 80(1), 78-88.
    Miyano et al. (2016). J. Phys. Chem. Lett., 7, 2199-2202.
    NREL Efficiency Chart (2024).
"""

import numpy as np


class PerovskitePVF1a:
    """Single-diode perovskite solar cell model (De Soto 5-parameter, n=1.5)."""

    def __init__(self, params: dict):
        mod = params["module"]
        ds = params["desoto_params"]

        self.cells_in_series = mod["cells_in_series"]["value"]
        self.alpha_sc = mod["alpha_sc"]["value"]
        self.area = mod["area"]["value"]           # m2
        self.p_mp_stc = mod["p_mp_stc"]["value"]  # W (for fallback)
        self.v_mp_stc = mod["v_mp_stc"]["value"]
        self.i_mp_stc = mod["i_mp_stc"]["value"]
        self.v_oc_stc = mod["v_oc_stc"]["value"]
        self.i_sc_stc = mod["i_sc_stc"]["value"]

        # De Soto reference parameters at STC
        self.I_L_ref  = ds["I_L_ref"]["value"]
        self.I_o_ref  = ds["I_o_ref"]["value"]
        self.R_s      = ds["R_s"]["value"]
        self.R_sh_ref = ds["R_sh_ref"]["value"]
        self.a_ref    = ds["a_ref"]["value"]
        self.EgRef    = ds["EgRef"]["value"]
        self.dEgdT    = ds["dEgdT"]["value"]

        # Physical constants
        self.k    = 1.380649e-23    # Boltzmann [J/K]
        self.q    = 1.602176634e-19 # electron charge [C]
        self.T_ref = 298.15         # STC temperature [K]
        self.G_ref = 1000.0         # STC irradiance [W/m2]

    def _calc_params(self, irradiance, cell_temp_c):
        """Calculate 5 diode parameters at given operating conditions."""
        G = np.asarray(irradiance, dtype=float)
        T = np.asarray(cell_temp_c, dtype=float) + 273.15  # to Kelvin

        # Light current — scales linearly with irradiance + small temp correction
        I_L = (G / self.G_ref) * (self.I_L_ref + self.alpha_sc * (T - self.T_ref))

        # Bandgap temperature dependence
        Eg = self.EgRef + self.dEgdT * (T - self.T_ref)

        # Saturation current — Boltzmann-weighted bandgap
        I_o = self.I_o_ref * (T / self.T_ref)**3 * np.exp(
            (self.EgRef / (self.k / self.q * self.T_ref)) -
            (Eg         / (self.k / self.q * T))
        )

        # Modified ideality factor scales with T
        a = self.a_ref * (T / self.T_ref)

        # Shunt resistance — inversely proportional to irradiance
        R_sh = self.R_sh_ref * (self.G_ref / np.maximum(G, 1.0))

        return I_L, I_o, self.R_s, R_sh, a

    def mpp(self, irradiance, cell_temp_c):
        """Maximum power point via pvlib singlediode or analytical fallback."""
        try:
            from pvlib.pvsystem import singlediode
            I_L, I_o, R_s, R_sh, a = self._calc_params(irradiance, cell_temp_c)
            result = singlediode(I_L, I_o, R_s, R_sh, a, method="lambertw")
            return {
                "v_mp":  np.asarray(result["v_mp"]),
                "i_mp":  np.asarray(result["i_mp"]),
                "p_mp":  np.asarray(result["p_mp"]),
                "v_oc":  np.asarray(result["v_oc"]),
                "i_sc":  np.asarray(result["i_sc"]),
            }
        except ImportError:
            return self._mpp_fallback(irradiance, cell_temp_c)

    def _mpp_fallback(self, irradiance, cell_temp_c):
        """
        Simplified MPP without pvlib.
        Uses linear irradiance scaling and affine temperature derating.
        Perovskite: Voc has slightly positive temp coeff at very low T,
        but overall ~-3 mV/K typical above 25C.
        """
        G = np.asarray(irradiance, dtype=float)
        T = np.asarray(cell_temp_c, dtype=float)
        dT = T - 25.0

        # p_mp_stc = 0.5W (Vmp=1.0V * Imp=0.5A for 25cm2 at 20% eff)
        p_mp = self.p_mp_stc * (G / 1000.0) * (1.0 - 0.003 * dT)
        p_mp = np.maximum(p_mp, 0.0)

        v_oc = self.v_oc_stc * (1.0 - 0.003 * dT)
        v_mp = self.v_mp_stc * (1.0 - 0.003 * dT)
        i_sc = self.i_sc_stc * (G / 1000.0) * (1.0 + 0.0003 * dT)
        i_mp = np.where(v_mp > 0, p_mp / np.maximum(v_mp, 0.01), 0.0)

        return {"v_mp": v_mp, "i_mp": i_mp, "p_mp": p_mp, "v_oc": v_oc, "i_sc": i_sc}

    def efficiency(self, irradiance, cell_temp_c):
        """Cell efficiency = P_mp / (G * area)."""
        result = self.mpp(irradiance, cell_temp_c)
        G = np.asarray(irradiance, dtype=float)
        return np.where(G > 0, result["p_mp"] / (G * self.area), 0.0)
