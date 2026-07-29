"""
EC027 — Solid-State Lithium Battery — F1a SOC-Only Model

Simple semi-empirical voltage model:
    V_terminal = OCV(SOC) - I * R_internal

OCV(SOC) is a 5th-order polynomial fit representing a layered-oxide
cathode paired with a Li-metal anode through a solid electrolyte.
The OCV window is similar to NMC but extends to a higher upper cut-off
voltage (~4.3 V) thanks to the wider electrochemical stability window
of sulfide / oxide solid electrolytes.

Reference:
    Janek, J., Zeier, W. G. (2016). "A solid future for battery
    development." Nature Energy, 1, 16141.
"""

import numpy as np


class SolidStateLiF1a:
    """Solid-state Li battery cell model — voltage as a function of SOC and current only."""

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
