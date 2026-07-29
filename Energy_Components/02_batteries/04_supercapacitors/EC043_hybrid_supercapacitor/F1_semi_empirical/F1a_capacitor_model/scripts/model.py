"""
EC043 — Hybrid Supercapacitor — F1a Capacitor Model

Simple RC model:
    V = Q / C + I * R_esr

V_max = 3.8 V, C = 200 F, R_esr = 10 mΩ
Energy stored: E = 0.5 * C * V_oc^2

References:
    Conway (1999). Electrochemical Supercapacitors: Scientific Fundamentals.
    IEC 62576 — Electric double-layer capacitors for use in hybrid electric vehicles.
"""

import numpy as np


class HybridSupercapacitorF1a:
    """RC capacitor model for hybrid supercapacitor."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.C = u["C"]["value"]          # F
        self.R_esr = u["R_esr"]["value"]  # Ohm
        self.V_max = u["V_max"]["value"]  # V
        self.V_min = u["V_min"]["value"]  # V
        self.Q_max = u["Q_max"]["value"]  # C = C_cap * V_max

    def predict(self, Q, I):
        """
        Parameters
        ----------
        Q : float or array, Coulombs — stored charge
        I : float or array, Amperes — current (positive = discharge)

        Returns
        -------
        dict with V_terminal, V_oc, SOC, P_W, E_Wh, V_drop_esr
        """
        Q = np.asarray(Q, dtype=float)
        I = np.asarray(I, dtype=float)

        Q_clipped = np.clip(Q, 0.0, self.Q_max)
        V_oc = Q_clipped / self.C                    # open-circuit voltage
        V_drop = I * self.R_esr                      # ESR voltage drop
        V_terminal = V_oc - V_drop                   # terminal voltage
        V_terminal = np.clip(V_terminal, 0.0, self.V_max)

        SOC = Q_clipped / self.Q_max                 # 0..1
        P_W = V_terminal * I                         # instantaneous power
        E_Wh = 0.5 * self.C * V_oc ** 2 / 3600.0   # stored energy

        return {
            "V_terminal": V_terminal,
            "V_oc": V_oc,
            "SOC": SOC,
            "P_W": P_W,
            "E_Wh": E_Wh,
            "V_drop_esr": V_drop,
        }
