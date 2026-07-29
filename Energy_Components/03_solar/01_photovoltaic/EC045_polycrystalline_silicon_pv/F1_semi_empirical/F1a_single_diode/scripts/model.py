"""
EC045 — Polycrystalline Silicon PV — F1a Single-Diode Model

Pure-NumPy implementation of the De Soto 5-parameter single-diode equation:

    I = I_L - I_o * [exp((V + I*R_s) / a) - 1] - (V + I*R_s) / R_sh

Translation of reference parameters to operating conditions:
    I_L  = (G/G_ref) * [I_L_ref + alpha_sc*(T - T_ref)]
    I_o  = I_o_ref * (T/T_ref)^3 * exp[(Eg_ref/(k/q*T_ref)) - (Eg/(k/q*T))]
    a    = a_ref * T / T_ref
    R_sh = R_sh_ref * (G_ref / G)
    R_s  = constant

The implicit single-diode equation is solved by Newton's method (no pvlib).

Reference:
    De Soto et al. (2006). "Improvement and validation of a model for photovoltaic
    array performance." Solar Energy, 80(1), 78-88.
"""

import numpy as np


class PolySiPVF1a:
    """Polycrystalline-Si PV — De Soto 5-parameter single-diode model (pure NumPy)."""

    def __init__(self, params: dict):
        mod = params["module"]
        ds = params["desoto_params"]

        self.cells_in_series = mod["cells_in_series"]["value"]
        self.alpha_sc = mod["alpha_sc"]["value"]
        self.area = mod.get("area", {"value": 1.638})["value"]

        self.I_L_ref = ds["I_L_ref"]["value"]
        self.I_o_ref = ds["I_o_ref"]["value"]
        self.R_s = ds["R_s"]["value"]
        self.R_sh_ref = ds["R_sh_ref"]["value"]
        self.a_ref = ds["a_ref"]["value"]
        self.EgRef = ds["EgRef"]["value"]
        self.dEgdT = ds["dEgdT"]["value"]

        self.k = 1.380649e-23
        self.q = 1.602176634e-19
        self.T_ref = 298.15
        self.G_ref = 1000.0

    # ------------------------------------------------------------------
    # Five-parameter translation
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
    # Implicit single-diode I(V) — Newton solver
    # ------------------------------------------------------------------
    def _i_from_v(self, V, I_L, I_o, R_sh, a, n_iter=40):
        V = np.asarray(V, dtype=float)
        I = np.asarray(I_L, dtype=float).copy()  # initial guess
        for _ in range(n_iter):
            arg = (V + I * self.R_s) / a
            arg = np.clip(arg, -50.0, 50.0)
            exp_term = np.exp(arg)
            f = I_L - I_o * (exp_term - 1.0) - (V + I * self.R_s) / R_sh - I
            df = -I_o * exp_term * (self.R_s / a) - self.R_s / R_sh - 1.0
            I = I - f / df
        return I

    # ------------------------------------------------------------------
    # Open-circuit voltage (I=0) — Newton solver
    # ------------------------------------------------------------------
    def _voc(self, I_L, I_o, R_sh, a, n_iter=40):
        # Start with ideal-diode approximation
        V = a * np.log(np.maximum(I_L / np.maximum(I_o, 1e-30), 1.0) + 1.0)
        for _ in range(n_iter):
            arg = np.clip(V / a, -50.0, 50.0)
            exp_term = np.exp(arg)
            f = I_L - I_o * (exp_term - 1.0) - V / R_sh
            df = -I_o * exp_term / a - 1.0 / R_sh
            V = V - f / df
        return np.maximum(V, 0.0)

    # ------------------------------------------------------------------
    # Maximum-power point — golden-section search on V
    # ------------------------------------------------------------------
    def mpp(self, irradiance, cell_temp_c):
        G = np.asarray(irradiance, dtype=float)
        T_c = np.asarray(cell_temp_c, dtype=float)
        scalar = (G.ndim == 0) and (T_c.ndim == 0)
        G_b = np.broadcast_to(G, np.broadcast_shapes(G.shape, T_c.shape)).astype(float)
        T_b = np.broadcast_to(T_c, G_b.shape).astype(float)

        I_L, I_o, R_sh, a = self._calc_params(G_b, T_b)
        V_oc = self._voc(I_L, I_o, R_sh, a)
        I_sc = self._i_from_v(np.zeros_like(V_oc), I_L, I_o, R_sh, a)

        # Golden-section search for V_mp on [0, V_oc]
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

        # Zero-irradiance handling
        zero = G_b <= 1.0
        V_mp = np.where(zero, 0.0, V_mp)
        I_mp = np.where(zero, 0.0, I_mp)
        P_mp = np.where(zero, 0.0, P_mp)
        V_oc = np.where(zero, 0.0, V_oc)
        I_sc = np.where(zero, 0.0, I_sc)

        out = {"v_mp": V_mp, "i_mp": I_mp, "p_mp": P_mp, "v_oc": V_oc, "i_sc": I_sc}
        if scalar:
            out = {k: np.asarray(v).reshape(()) for k, v in out.items()}
        return out

    def efficiency(self, irradiance, cell_temp_c):
        result = self.mpp(irradiance, cell_temp_c)
        G = np.asarray(irradiance, dtype=float)
        return np.where(G > 1.0, result["p_mp"] / (np.maximum(G, 1.0) * self.area), 0.0)
