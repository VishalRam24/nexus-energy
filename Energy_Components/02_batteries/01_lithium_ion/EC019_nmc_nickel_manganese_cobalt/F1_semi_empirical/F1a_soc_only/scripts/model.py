"""
EC019 — NMC Battery — F1a SOC-Only Model

Simple semi-empirical voltage model:
    V_terminal = OCV(SOC) - I * R_internal

where OCV(SOC) is a 5th-order polynomial fit to NMC811 open-circuit voltage data.

Reference:
    Chen et al. (2020). "Development of Experimental Techniques for Parameterization
    of Multi-scale Lithium-ion Battery Models." J. Electrochem. Soc., 167, 080534.
"""

import numpy as np


class NMCBatteryF1a:
    """NMC battery cell model — voltage as a function of SOC and current only."""

    def __init__(self, params: dict):
        cell = params["cell"]
        ocv = params["ocv_coefficients"]

        self.capacity = cell["capacity"]["value"]          # Ah
        self.v_max = cell["voltage_max"]["value"]           # V
        self.v_min = cell["voltage_min"]["value"]           # V
        self.r_internal = cell["internal_resistance"]["value"]  # Ohm

        # OCV polynomial coefficients (order 0 to 5)
        self.ocv_coeff = np.array([ocv[f"a{i}"] for i in range(6)])

    def ocv(self, soc):
        """Open-circuit voltage as a function of SOC (0-1)."""
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([soc**i for i in range(6)], axis=-1)
        return np.dot(powers, self.ocv_coeff)

    def terminal_voltage(self, soc, current):
        """
        Terminal voltage.

        Args:
            soc: State of charge (0-1)
            current: Current in A (positive = discharge, negative = charge)

        Returns:
            Voltage in V (clipped to [v_min, v_max])
        """
        soc = np.asarray(soc, dtype=float)
        current = np.asarray(current, dtype=float)
        v = self.ocv(soc) - current * self.r_internal
        return np.clip(v, self.v_min, self.v_max)

    def power(self, soc, current):
        """Electrical power in W (positive = discharging, negative = charging)."""
        return self.terminal_voltage(soc, current) * current

    def soc_derivative(self, soc, current):
        """dSOC/dt in 1/s. Integrate to get SOC trajectory."""
        return -current / (self.capacity * 3600.0)
