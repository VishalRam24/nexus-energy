"""
EC018 — LFP Battery — F1a SOC-Only Model

Simple semi-empirical voltage model:
    V_terminal = OCV(SOC) - I * R_internal

LFP characteristic: very flat voltage plateau around 3.3V between 20-80% SOC.

Reference:
    Chen et al. (2020). J. Electrochem. Soc., 167, 080534.
    A123 Systems ANR26650M1B datasheet.
"""

import numpy as np


class LFPBatteryF1a:
    """LFP battery cell model — voltage as a function of SOC and current only."""

    def __init__(self, params: dict):
        cell = params["cell"]
        ocv = params["ocv_coefficients"]

        self.capacity = cell["capacity"]["value"]
        self.v_max = cell["voltage_max"]["value"]
        self.v_min = cell["voltage_min"]["value"]
        self.r_internal = cell["internal_resistance"]["value"]

        self.ocv_coeff = np.array([ocv[f"a{i}"] for i in range(6)])

    def ocv(self, soc):
        """Open-circuit voltage as a function of SOC (0-1)."""
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([soc**i for i in range(6)], axis=-1)
        return np.dot(powers, self.ocv_coeff)

    def terminal_voltage(self, soc, current):
        """Terminal voltage. current: positive=discharge, negative=charge."""
        soc = np.asarray(soc, dtype=float)
        current = np.asarray(current, dtype=float)
        v = self.ocv(soc) - current * self.r_internal
        return np.clip(v, self.v_min, self.v_max)

    def power(self, soc, current):
        """Electrical power in W."""
        return self.terminal_voltage(soc, current) * current

    def soc_derivative(self, soc, current):
        """dSOC/dt in 1/s."""
        return -current / (self.capacity * 3600.0)
