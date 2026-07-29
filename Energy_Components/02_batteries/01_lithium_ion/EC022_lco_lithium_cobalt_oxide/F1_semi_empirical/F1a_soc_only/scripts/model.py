"""
EC022 — LCO (Lithium Cobalt Oxide) Battery — F1a SOC-Only Model

Simple semi-empirical voltage model:
    V_terminal = OCV(SOC) - I * R_internal

where OCV(SOC) is a 5th-order polynomial fit to a representative LCO
discharge curve. LCO is the original Li-ion chemistry used in consumer
electronics, with a nominal 3.7 V plateau.

Reference:
    Reimers, J. N., Dahn, J. R. (1992). "Electrochemical and In Situ
    X-Ray Diffraction Studies of Lithium Intercalation in LixCoO2."
    J. Electrochem. Soc., 139, 2091.
"""

import numpy as np


class LCOBatteryF1a:
    """LCO battery cell model — voltage as a function of SOC and current only."""

    def __init__(self, params: dict):
        cell = params["cell"]
        ocv = params["ocv_coefficients"]

        self.capacity = cell["capacity"]["value"]
        self.v_max = cell["voltage_max"]["value"]
        self.v_min = cell["voltage_min"]["value"]
        self.r_internal = cell["internal_resistance"]["value"]
        self.ocv_coeff = np.array([ocv[f"a{i}"] for i in range(6)])

    def ocv(self, soc):
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([soc**i for i in range(6)], axis=-1)
        return np.dot(powers, self.ocv_coeff)

    def terminal_voltage(self, soc, current):
        soc = np.asarray(soc, dtype=float)
        current = np.asarray(current, dtype=float)
        v = self.ocv(soc) - current * self.r_internal
        return np.clip(v, self.v_min, self.v_max)

    def power(self, soc, current):
        return self.terminal_voltage(soc, current) * current

    def soc_derivative(self, soc, current):
        return -current / (self.capacity * 3600.0)
