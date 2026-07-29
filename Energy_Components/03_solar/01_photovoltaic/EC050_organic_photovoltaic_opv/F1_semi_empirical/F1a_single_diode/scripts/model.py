"""
EC050 — Organic Photovoltaic (OPV) — F1a Single-Diode Model

5-parameter single-diode model (per unit area, scaled by A_cell):
    I = I_L - I0*(exp((V + I*Rs)/(n*Vt)) - 1) - (V + I*Rs)/Rsh

Solved implicitly via Newton-Raphson for the I-V curve.
MPP found by scanning the I-V curve.

Parameters (typical BHJ OPV):
    I_L = 8 mA/cm², I0 = 1e-6 A/cm², n = 2.0
    Rs = 10 Ω·cm², Rsh = 500 Ω·cm², Voc ≈ 0.8 V, A = 100 cm²

References:
    De Soto et al. (2006). Improvement and validation of a model for PV array performance.
    Brabec et al. (2001). Plastic Solar Cells. Adv. Funct. Mater.
"""

import numpy as np


class OPVSingleDiodeF1a:
    """Single-diode model for OPV cell."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.I_L0 = u["I_L_density"]["value"]   # A/cm²
        self.I0 = u["I0_density"]["value"]       # A/cm²
        self.n = u["n"]["value"]
        self.Rs = u["Rs_area"]["value"]          # Ω·cm²
        self.Rsh = u["Rsh_area"]["value"]        # Ω·cm²
        self.A = u["A_cell"]["value"]            # cm²
        self.Vt = u["Vt"]["value"]               # V

    def _current_density(self, V_arr, J_arr, G_frac):
        """Implicit equation: J - J_L + I0*(exp(...)-1) + (V+J*Rs)/Rsh = 0"""
        nVt = self.n * self.Vt
        arg = (V_arr + J_arr * self.Rs) / nVt
        arg = np.clip(arg, -50, 50)
        return (G_frac * self.I_L0
                - self.I0 * (np.exp(arg) - 1.0)
                - (V_arr + J_arr * self.Rs) / self.Rsh)

    def _solve_iv(self, V_pts, G_frac):
        """Solve I(V) via Newton-Raphson for each voltage point."""
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
        """
        Parameters
        ----------
        G : irradiance W/m2 (1000 = STC)
        n_points : number of I-V curve points

        Returns
        -------
        dict with Voc, Isc, Vmp, Imp, Pmp, FF, eta, I_curve, V_curve
        """
        G = float(G)
        G_frac = G / 1000.0

        I_L = G_frac * self.I_L0  # A/cm²

        # Voc estimate: V such that I=0 → scan
        V_pts = np.linspace(0.0, 1.2, n_points)
        J_arr = self._solve_iv(V_pts, G_frac)

        # Find Voc (where J crosses zero)
        sign_change = np.where(np.diff(np.sign(J_arr)))[0]
        if len(sign_change) > 0:
            idx = sign_change[0]
            # linear interpolation
            Voc = V_pts[idx] - J_arr[idx] * (V_pts[idx+1] - V_pts[idx]) / (J_arr[idx+1] - J_arr[idx])
        else:
            Voc = V_pts[np.argmin(np.abs(J_arr))]

        Voc = float(np.clip(Voc, 0.0, 1.2))

        # Clip curve to [0, Voc]
        mask = V_pts <= Voc
        V_curve = V_pts[mask]
        J_curve = np.maximum(J_arr[mask], 0.0)

        Isc = float(G_frac * self.I_L0)  # at V=0
        P_curve = V_curve * J_curve
        idx_mpp = int(np.argmax(P_curve))
        Vmp = float(V_curve[idx_mpp])
        Imp = float(J_curve[idx_mpp])
        Pmp = float(P_curve[idx_mpp])  # W/cm²

        FF = Pmp / (Voc * Isc) if (Voc * Isc) > 0 else 0.0
        eta = Pmp / (G / 1e4) if G > 0 else 0.0  # G in W/cm²

        # Scale to full cell
        return {
            "Voc_V": Voc,
            "Isc_A": Isc * self.A,
            "Vmp_V": Vmp,
            "Imp_A": Imp * self.A,
            "Pmp_W": Pmp * self.A,
            "FF": FF,
            "eta": eta,
        }
