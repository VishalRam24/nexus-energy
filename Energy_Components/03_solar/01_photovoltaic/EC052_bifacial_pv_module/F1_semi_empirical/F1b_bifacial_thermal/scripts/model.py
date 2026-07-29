"""
EC052 — Bifacial PV Module — F1b Bifacial + Thermal Model

Extends F1a (bifacial gain + single-diode) with:
  1. Separate front and rear cell temperature models
  2. Rear cell runs slightly cooler than front (rear_thermal_factor < 1)
  3. Effective cell temperature = irradiance-weighted average of front and rear

Physics:
    G_rear = bifaciality_factor * albedo * G_front * rear_view_factor
    G_eff  = G_front + G_rear                   (effective irradiance on equivalent monofacial)

    dT_front = G_front * (NOCT_front - T_NOCT_amb) / G_NOCT
    dT_rear  = G_rear  * rear_thermal_factor * (NOCT_front - T_NOCT_amb) / G_NOCT

    T_cell_front = T_amb + dT_front
    T_cell_rear  = T_amb + dT_rear               (cooler because less irradiance)

    T_cell_eff = (G_front * T_cell_front + G_rear * T_cell_rear) / max(G_eff, 1)
                                                 (irradiance-weighted average)

    I-V via De Soto 5-parameter at (G_eff, T_cell_eff)

References:
    Deline et al. (2017). "A simplified model of uniform shading in bifacial PV systems."
    IEEE PVSC.
    Marion et al. (2017). "New approaches for modeling bifacial modules in PVsyst."
    IEEE JPV 7(6), 1519-1526.
    Cvetkovska et al. (2021). "Analysis of thermal behaviour of bifacial PV modules."
    IEEE JPV 11(4), 938-944.
"""

import numpy as np


class BifacialPVF1b:
    """Bifacial PV module — single-diode (De Soto) + front/rear thermal model."""

    def __init__(self, params: dict):
        mod = params["module"]
        ds = params["desoto_params"]

        self.alpha_sc = mod["alpha_sc"]["value"]
        self.gamma_pmp = mod["gamma_pmp"]["value"]
        self.cells_in_series = mod["cells_in_series"]["value"]
        self.area = mod["area"]["value"]
        self.bifaciality = mod["bifaciality_factor"]["value"]
        self.rear_view_factor = mod["rear_view_factor"]["value"]

        # Thermal
        self.NOCT_front = mod["NOCT_front"]["value"]
        self.T_NOCT_amb = mod["T_NOCT_amb"]["value"]
        self.G_NOCT = mod["G_NOCT"]["value"]
        self.rear_thermal_factor = mod["rear_thermal_factor"]["value"]

        # De Soto
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
    # Bifacial irradiance and temperature models
    # ------------------------------------------------------------------
    def effective_irradiance(self, G_front, albedo=0.2, G_rear=None):
        """
        Effective irradiance for equivalent monofacial model.
        G_rear can be supplied directly or computed from albedo.
        """
        G_f = np.asarray(G_front, dtype=float)
        if G_rear is not None:
            G_r = np.asarray(G_rear, dtype=float)
        else:
            alb = np.asarray(albedo, dtype=float)
            G_r = self.bifaciality * alb * G_f * self.rear_view_factor
        return G_f + G_r, G_r

    def cell_temperatures(self, G_front, T_amb_c, albedo=0.2, G_rear=None):
        """
        Front and rear cell temperatures.
        Rear is slightly cooler because it receives less irradiance.
        Returns T_cell_front, T_cell_rear, T_cell_eff (irradiance-weighted mean)
        """
        G_f = np.asarray(G_front, dtype=float)
        T_amb = np.asarray(T_amb_c, dtype=float)
        G_eff, G_r = self.effective_irradiance(G_f, albedo, G_rear)

        dT_noct = (self.NOCT_front - self.T_NOCT_amb) / self.G_NOCT
        T_front = T_amb + G_f * dT_noct
        T_rear = T_amb + G_r * dT_noct * self.rear_thermal_factor

        # Irradiance-weighted effective cell temperature
        T_eff = np.where(
            G_eff > 1.0,
            (G_f * T_front + G_r * T_rear) / np.maximum(G_eff, 1.0),
            T_amb,
        )
        return T_front, T_rear, T_eff

    # ------------------------------------------------------------------
    # De Soto 5-parameter solver
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

    def _mpp_solve(self, G_eff, T_cell_eff):
        G = np.asarray(G_eff, dtype=float)
        T_c = np.asarray(T_cell_eff, dtype=float)
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
        out = {"v_mp": V_mp, "i_mp": I_mp, "p_mp": P_mp,
               "v_oc": V_oc, "i_sc": I_sc, "fill_factor": fill_factor}
        if scalar:
            out = {k: np.asarray(v).reshape(()) for k, v in out.items()}
        return out

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def mpp(self, G_front, T_amb_c, albedo=0.2, G_rear=None):
        """
        Full bifacial + thermal MPP calculation.

        Parameters
        ----------
        G_front : W/m2, front irradiance
        T_amb_c : degC, ambient temperature
        albedo  : ground/surface albedo (used if G_rear not supplied)
        G_rear  : W/m2, rear irradiance (optional direct override)
        """
        G_eff, G_r = self.effective_irradiance(G_front, albedo, G_rear)
        T_front, T_rear, T_eff = self.cell_temperatures(G_front, T_amb_c, albedo, G_rear)
        result = self._mpp_solve(G_eff, T_eff)
        result["G_eff_w_m2"] = np.asarray(G_eff)
        result["G_rear_w_m2"] = np.asarray(G_r)
        result["T_cell_front_c"] = np.asarray(T_front)
        result["T_cell_rear_c"] = np.asarray(T_rear)
        result["T_cell_eff_c"] = np.asarray(T_eff)
        return result

    def efficiency(self, G_front, T_amb_c, albedo=0.2, G_rear=None):
        result = self.mpp(G_front, T_amb_c, albedo, G_rear)
        G_f = np.asarray(G_front, dtype=float)
        return np.where(G_f > 1.0,
                        result["p_mp"] / (np.maximum(G_f, 1.0) * self.area), 0.0)

    def bifacial_gain(self, G_front, T_amb_c, albedo=0.2, G_rear=None):
        """Bifacial gain = (P_bifacial - P_front_only) / P_front_only."""
        p_bifacial = self.mpp(G_front, T_amb_c, albedo, G_rear)["p_mp"]
        p_front = self._mpp_solve(
            np.asarray(G_front, dtype=float),
            np.asarray(T_amb_c, dtype=float) + np.asarray(G_front, dtype=float)
            * (self.NOCT_front - self.T_NOCT_amb) / self.G_NOCT,
        )["p_mp"]
        return np.where(p_front > 0, (p_bifacial - p_front) / p_front, 0.0)
