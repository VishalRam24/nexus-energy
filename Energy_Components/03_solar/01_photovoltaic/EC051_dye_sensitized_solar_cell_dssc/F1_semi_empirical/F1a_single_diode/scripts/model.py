"""
EC051 — Dye-Sensitized Solar Cell (DSSC) — F1a Single-Diode Model

5-parameter single-diode model for DSSC:
    I = I_L - I0*(exp((V + I*Rs)/(n*Vt)) - 1) - (V + I*Rs)/Rsh

DSSC characteristics: n=2, I_L=15 mA/cm², Voc~0.7V, η~8-11%

References:
    O'Regan & Graetzel (1991). A low-cost, high-efficiency solar cell based on dye-sensitized
        colloidal TiO2 films. Nature.
    Grätzel (2003). Dye-sensitized solar cells. J. Photochem. Photobiol. C.
"""

import numpy as np


class DSSCSingleDiodeF1a:
    """Single-diode model for DSSC."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.I_L0 = u["I_L_density"]["value"]
        self.I0 = u["I0_density"]["value"]
        self.n = u["n"]["value"]
        self.Rs = u["Rs_area"]["value"]
        self.Rsh = u["Rsh_area"]["value"]
        self.A = u["A_cell"]["value"]
        self.Vt = u["Vt"]["value"]

    def _solve_iv(self, V_pts, G_frac):
        nVt = self.n * self.Vt
        J = np.full_like(V_pts, G_frac * self.I_L0)
        for _ in range(50):
            arg = np.clip((V_pts + J * self.Rs) / nVt, -50, 50)
            exp_arg = np.exp(arg)
            F = J - G_frac * self.I_L0 + self.I0 * (exp_arg - 1.0) + (V_pts + J * self.Rs) / self.Rsh
            dF = 1.0 + self.I0 * exp_arg * self.Rs / nVt + self.Rs / self.Rsh
            J -= F / dF
        return np.maximum(J, 0.0)

    def predict(self, G=1000.0, n_points=200):
        G = float(G)
        G_frac = G / 1000.0

        V_pts = np.linspace(0.0, 1.0, n_points)
        J_arr = self._solve_iv(V_pts, G_frac)

        sign_change = np.where(np.diff(np.sign(J_arr)))[0]
        if len(sign_change) > 0:
            idx = sign_change[0]
            Voc = V_pts[idx] - J_arr[idx] * (V_pts[idx+1] - V_pts[idx]) / (J_arr[idx+1] - J_arr[idx])
        else:
            Voc = V_pts[np.argmin(np.abs(J_arr))]
        Voc = float(np.clip(Voc, 0.0, 1.0))

        mask = V_pts <= Voc
        V_curve = V_pts[mask]
        J_curve = np.maximum(J_arr[mask], 0.0)
        P_curve = V_curve * J_curve

        Isc = float(G_frac * self.I_L0)
        idx_mpp = int(np.argmax(P_curve))
        Vmp = float(V_curve[idx_mpp])
        Imp = float(J_curve[idx_mpp])
        Pmp = float(P_curve[idx_mpp])  # W/cm²

        FF = Pmp / (Voc * Isc) if (Voc * Isc) > 0 else 0.0
        eta = Pmp / (G / 1e4) if G > 0 else 0.0

        return {
            "Voc_V": Voc,
            "Isc_A": Isc * self.A,
            "Vmp_V": Vmp,
            "Imp_A": Imp * self.A,
            "Pmp_W": Pmp * self.A,
            "FF": FF,
            "eta": eta,
        }
