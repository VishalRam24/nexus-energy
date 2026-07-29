"""
EC035 — Sodium-Sulfur (NaS) Battery — F1a SOC-Only Model

Simple semi-empirical voltage model:
    V_terminal = OCV(SOC) - I * R_internal

OCV(SOC) is a 5th-order polynomial fit to a representative NaS
discharge curve. NaS cells operate at ~300-350 C (molten Na, molten S
and Na polysulfides separated by a beta"-alumina solid electrolyte) and
exhibit a two-plateau OCV between roughly 1.78 V and 2.08 V per cell.

Reference:
    Wen, Z., Hu, Y., Wu, X., Han, J., Gu, Z. (2008). "Main challenges
    for high performance NAS battery: materials and interfaces."
    Materials Science and Engineering B, 154-155, 73.
"""

import numpy as np


class NaSBatteryF1a:
    """NaS battery cell model — voltage as a function of SOC and current only."""

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
