"""
EC052 — Bifacial PV Module — F1a Bifacial Gain + Single-Diode Model

Effective irradiance:
    G_rear_used = G_rear           if user provides it directly,
                  albedo * G_front * F_view  otherwise.
    G_eff = G_front + phi * G_rear_used,
    phi   = bifaciality factor (rear/front nameplate ratio, typ 0.7-0.85).

The single-diode equation (De Soto 5-parameter, pure NumPy) is then solved with
G = G_eff. Module rating remains based on the front-side STC nameplate.

Reference:
    De Soto et al. (2006), Solar Energy 80(1).
    Stein et al. (2017), 'Bifacial PV Modules: A Review of Modeling Techniques',
    Sandia National Laboratories.
"""

import numpy as np


class BifacialPVF1a:
    """Bifacial PV — single-diode model with rear-side gain (pure NumPy)."""

    def __init__(self, params: dict):
        mod = params["module"]
        ds = params["desoto_params"]

        self.cells_in_series = mod["cells_in_series"]["value"]
        self.alpha_sc = mod["alpha_sc"]["value"]
        self.area = mod.get("area", {"value": 2.0})["value"]
        self.phi = mod["bifaciality_factor"]["value"]
        self.F_view = mod["rear_view_factor"]["value"]

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
    # Bifacial effective irradiance
    # ------------------------------------------------------------------
    def effective_irradiance(self, G_front, G_rear=None, albedo=None):
        G_f = np.asarray(G_front, dtype=float)
        if G_rear is not None:
            G_r = np.asarray(G_rear, dtype=float)
        elif albedo is not None:
            a = np.asarray(albedo, dtype=float)
            G_r = a * G_f * self.F_view
        else:
            G_r = np.zeros_like(G_f)
        return G_f + self.phi * G_r, G_r

    # ------------------------------------------------------------------
    # Single-diode parameter translation
    # ------------------------------------------------------------------
    def _calc_params(self, G_eff, cell_temp_c):
        G = np.asarray(G_eff, dtype=float)
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

    def _i_from_v(self, V, I_L, I_o, R_sh, a, n_iter=40):
        V = np.asarray(V, dtype=float)
        I = np.asarray(I_L, dtype=float).copy()
        for _ in range(n_iter):
            arg = (V + I * self.R_s) / a
            arg = np.clip(arg, -50.0, 50.0)
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

    def mpp(self, G_front, cell_temp_c, G_rear=None, albedo=None):
        G_eff, G_rear_used = self.effective_irradiance(G_front, G_rear=G_rear, albedo=albedo)
        G_eff = np.asarray(G_eff, dtype=float)
        T_c = np.asarray(cell_temp_c, dtype=float)
        scalar = (G_eff.ndim == 0) and (T_c.ndim == 0)
        G_b = np.broadcast_to(G_eff, np.broadcast_shapes(G_eff.shape, T_c.shape)).astype(float)
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

        out = {
            "v_mp": V_mp, "i_mp": I_mp, "p_mp": P_mp,
            "v_oc": V_oc, "i_sc": I_sc,
            "G_effective": G_b,
            "G_rear_used": np.broadcast_to(np.asarray(G_rear_used, dtype=float), G_b.shape).copy(),
        }
        if scalar:
            out = {k: np.asarray(v).reshape(()) for k, v in out.items()}
        return out

    def efficiency(self, G_front, cell_temp_c, G_rear=None, albedo=None):
        result = self.mpp(G_front, cell_temp_c, G_rear=G_rear, albedo=albedo)
        G_f = np.asarray(G_front, dtype=float)
        return np.where(G_f > 1.0, result["p_mp"] / (np.maximum(G_f, 1.0) * self.area), 0.0)

    def bifacial_gain(self, G_front, cell_temp_c, G_rear=None, albedo=None):
        """(P_bifacial - P_front_only) / P_front_only."""
        r_bi = self.mpp(G_front, cell_temp_c, G_rear=G_rear, albedo=albedo)
        r_mono = self.mpp(G_front, cell_temp_c, G_rear=0.0, albedo=0.0)
        P_mono = np.asarray(r_mono["p_mp"], dtype=float)
        P_bi = np.asarray(r_bi["p_mp"], dtype=float)
        return np.where(P_mono > 1.0, (P_bi - P_mono) / np.maximum(P_mono, 1.0), 0.0)
