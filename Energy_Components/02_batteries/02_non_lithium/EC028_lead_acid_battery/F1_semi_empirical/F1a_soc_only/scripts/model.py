"""
EC028 — Lead-Acid Battery — F1a SOC-Voltage Model

Terminal voltage using modified Shepherd model:
    OCV(SOC) = a0 + a1*SOC + a2*SOC^2 + a3*SOC^3
    V        = OCV(SOC) - I * R_internal

Conventions:
    current > 0  =>  discharge  (voltage drops)
    current < 0  =>  charge     (voltage rises)

References:
    Copetti, J.B., Lorenzo, E., Chenlo, F. (1993).
    A general battery model for PV system simulation.
    Progress in Photovoltaics, 1(4), 283-292.

    Manwell, J.F., McGowan, J.G. (1993).
    Lead acid battery storage model for hybrid energy systems.
    Solar Energy, 50(5), 399-405.
"""

import numpy as np


class LeadAcidF1a:
    """Lead-acid battery terminal voltage as f(SOC, I)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V_nom      = u["V_nom"]["value"]          # V
        self.C_ah       = u["C_ah"]["value"]           # Ah
        self.V_min      = u["V_min"]["value"]          # V
        self.V_max      = u["V_max"]["value"]          # V
        self.R_int      = u["R_internal"]["value"]     # Ohm
        c = u["ocv_coeffs"]
        self.a0 = c["a0"]["value"]
        self.a1 = c["a1"]["value"]
        self.a2 = c["a2"]["value"]
        self.a3 = c["a3"]["value"]

    def ocv(self, soc):
        """Open-circuit voltage [V] as a cubic polynomial of SOC."""
        soc = np.asarray(soc, dtype=float)
        return self.a0 + self.a1 * soc + self.a2 * soc ** 2 + self.a3 * soc ** 3

    def voltage(self, soc, current):
        """Terminal voltage [V]. current > 0 = discharge."""
        soc     = np.asarray(soc,     dtype=float)
        current = np.asarray(current, dtype=float)
        return self.ocv(soc) - current * self.R_int

    def power_w(self, soc, current):
        """Terminal power [W]. Positive = discharge."""
        return self.voltage(soc, current) * np.asarray(current, dtype=float)

    def dsoc_dt(self, current):
        """Rate of change of SOC [1/s]. Positive current => negative dsoc/dt."""
        current = np.asarray(current, dtype=float)
        C_s = self.C_ah * 3600.0   # Ah -> As (Coulombs)
        return -current / C_s
