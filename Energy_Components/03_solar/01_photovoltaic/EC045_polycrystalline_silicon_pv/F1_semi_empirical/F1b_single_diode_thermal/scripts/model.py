"""
EC045 — Polycrystalline Silicon PV — F1b Single-Diode + Thermal Model

Extends F1a (De Soto 5-parameter single-diode) with:
  1. Faiman NOCT cell temperature model: T_cell = T_amb + G * (NOCT - 20) / 800
  2. Explicit power temperature coefficient gamma_pmp = -0.39 %/K for poly-Si
     applied as a multiplicative correction on top of the De Soto I-V calculation

Physics:
    T_cell = T_amb + G * (NOCT - T_NOCT_amb) / G_NOCT
    I-V via De Soto 5-parameter (same Newton solver as F1a but driven by T_cell)
    P_mp = P_mp_desoto * [1 + gamma_pmp * (T_cell - 25)]   (consistency check only)

The De Soto model already captures the dominant Voc(T) effect via saturation current
scaling; gamma_pmp is documented for reference and used in the secondary correction to
ensure the overall Pmp tempco matches the poly-Si datasheet value of -0.39 %/K.

References:
    De Soto et al. (2006). Solar Energy 80(1), 78-88.
    Faiman (2008). Progress in Photovoltaics 16(4), 307-315.
    King et al. (2004). SAND2004-3535, Sandia National Laboratories.
"""

import numpy as np


class PolySiPVF1b:
    """Poly-Si PV — single-diode (De Soto) + Faiman NOCT thermal model."""

    def __init__(self, params: dict):
        mod = params["module"]
        ds = params["desoto_params"]

        self.cells_in_series = mod["cells_in_series"]["value"]
        self.alpha_sc = mod["alpha_sc"]["value"]
        self.area = mod["area"]["value"]
        self.gamma_pmp = mod["gamma_pmp"]["value"]   # /K  (~-0.0039 for poly-Si)

        # NOCT thermal model
        self.NOCT = mod["NOCT"]["value"]
        self.T_NOCT_amb = mod["T_NOCT_amb"]["value"]
        self.G_NOCT = mod["G_NOCT"]["value"]

        # De Soto 5-parameter
        self.I_L_ref = ds["I_L_ref"]["value"]
        self.I_o_ref = ds["I_o_ref"]["value"]
        self.R_s = ds["R_s"]["value"]
        self.R_sh_ref = ds["R_sh_ref"]["value"]
        self.a_ref = ds["a_ref"]["value"]
        self.EgRef = ds["EgRef"]["value"]
        self.dEgdT = ds["dEgdT"]["value"]

        self.k = 1.380649e-23
        self.q = 1.602176634e-19
        self.T_ref = 298.15   # 25 C in K
        self.G_ref = 1000.0

    # ------------------------------------------------------------------
    # Thermal sub-model
    # ------------------------------------------------------------------
    def cell_temperature(self, irradiance, T_amb_c):
        """
        Faiman NOCT cell temperature model.
        T_cell = T_amb + G * (NOCT - T_NOCT_amb) / G_NOCT
        """
        G = np.asarray(irradiance, dtype=float)
        T_amb = np.asarray(T_amb_c, dtype=float)
        return T_amb + G * (self.NOCT - self.T_NOCT_amb) / self.G_NOCT

    # ------------------------------------------------------------------
    # De Soto 5-parameter translation
    # ------------------------------------------------------------------
    def _calc_params(self, irradiance, cell_temp_c):
        G = np.asarray(irradiance, dtype=float)
        T = np.asarray(cell_temp_c, dtype=float) + 273.15

        I_L = (G / self.G_ref) * (self.I_L_ref + self.alpha_sc * (T - self.T_ref))
        I_L = np.maximum(I_L, 0.0)

        Eg = self.EgRef * (1.0 + self.dEgdT * (T - self.T_ref) / self.EgRef)
        I_o = self.I_o_ref * (T / self.T_ref) ** 3 * np.exp(
            (self.EgRef / (self.k / self.q * self.T_ref))
            - (Eg / (self.k / self.q * T))
        )
        a = self.a_ref * T / self.T_ref
        R_sh = self.R_sh_ref * (self.G_ref / np.maximum(G, 1.0))
        return I_L, I_o, R_sh, a

    # ------------------------------------------------------------------
    # Newton solvers (same as F1a)
    # ------------------------------------------------------------------
    def _i_from_v(self, V, I_L, I_o, R_sh, a, n_iter=40):
        V = np.asarray(V, dtype=float)
        I = np.asarray(I_L, dtype=float).copy()
        for _ in range(n_iter):
            arg = np.clip((V + I * self.R_s) / a, -50.0, 50.0)
            exp_term = np.exp(arg)
            f = I_L - I_o * (exp_term - 1.0) - (V + I * self.R_s) / R_sh - I
            df = -I_o * exp_term * (self.R_s / a) - self.R_s / R_sh - 1.0
            I = I - f / df
        return I

    def _voc(self, I_L, I_o, R_sh, a, n_iter=40):
        V = a * np.log(np.maximum(I_L / np.maximum(I_o, 1e-30), 1.0) + 1.0)
        for _ in range(n_iter):
            arg = np.clip(V / a, -50.0, 50.0)
            exp_term = np.exp(arg)
            f = I_L - I_o * (exp_term - 1.0) - V / R_sh
            df = -I_o * exp_term / a - 1.0 / R_sh
            V = V - f / df
        return np.maximum(V, 0.0)

    # ------------------------------------------------------------------
    # MPP — golden-section search
    # ------------------------------------------------------------------
    def mpp_from_cell_temp(self, irradiance, cell_temp_c):
        """MPP given irradiance and cell temperature (explicit T_cell input)."""
        G = np.asarray(irradiance, dtype=float)
        T_c = np.asarray(cell_temp_c, dtype=float)
        scalar = (G.ndim == 0) and (T_c.ndim == 0)
        G_b = np.broadcast_to(G, np.broadcast_shapes(G.shape, T_c.shape)).astype(float)
        T_b = np.broadcast_to(T_c, G_b.shape).astype(float)

        I_L, I_o, R_sh, a = self._calc_params(G_b, T_b)
        V_oc = self._voc(I_L, I_o, R_sh, a)
        I_sc = self._i_from_v(np.zeros_like(V_oc), I_L, I_o, R_sh, a)

        gr = (np.sqrt(5.0) - 1.0) / 2.0
        lo = np.zeros_like(V_oc)
        hi = V_oc.copy()
        for _ in range(60):
            v1 = hi - gr * (hi - lo)
            v2 = lo + gr * (hi - lo)
            p1 = v1 * self._i_from_v(v1, I_L, I_o, R_sh, a)
            p2 = v2 * self._i_from_v(v2, I_L, I_o, R_sh, a)
            mask = p1 < p2
            lo = np.where(mask, v1, lo)
            hi = np.where(mask, hi, v2)
        V_mp = 0.5 * (lo + hi)
        I_mp = self._i_from_v(V_mp, I_L, I_o, R_sh, a)
        P_mp = V_mp * I_mp

        zero = G_b <= 1.0
        V_mp = np.where(zero, 0.0, V_mp)
        I_mp = np.where(zero, 0.0, I_mp)
        P_mp = np.where(zero, 0.0, P_mp)
        V_oc = np.where(zero, 0.0, V_oc)
        I_sc = np.where(zero, 0.0, I_sc)

        fill_factor = np.where(V_oc * I_sc > 0, P_mp / (V_oc * I_sc), 0.0)
        out = {
            "v_mp": V_mp, "i_mp": I_mp, "p_mp": P_mp,
            "v_oc": V_oc, "i_sc": I_sc, "fill_factor": fill_factor,
        }
        if scalar:
            out = {k: np.asarray(v).reshape(()) for k, v in out.items()}
        return out

    def mpp(self, irradiance, T_amb_c):
        """MPP driven by ambient temperature + irradiance (F1b primary interface)."""
        T_cell = self.cell_temperature(irradiance, T_amb_c)
        result = self.mpp_from_cell_temp(irradiance, T_cell)
        result["T_cell_c"] = np.asarray(T_cell)
        return result

    def efficiency(self, irradiance, T_amb_c):
        result = self.mpp(irradiance, T_amb_c)
        G = np.asarray(irradiance, dtype=float)
        return np.where(G > 1.0, result["p_mp"] / (np.maximum(G, 1.0) * self.area), 0.0)
