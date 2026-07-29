"""
EC020 — NCA Battery (Nickel-Cobalt-Aluminum) — F1a SOC-Only Model

Simple semi-empirical voltage model:
    V_terminal = OCV(SOC) - I * R_internal

where OCV(SOC) is a 5th-order polynomial fitted to NCA (Panasonic NCR18650B) characteristics:
    OCV = a0 + a1*SOC + a2*SOC^2 + a3*SOC^3 + a4*SOC^4 + a5*SOC^5

Coefficients give:
    OCV(0.0) ≈ 2.7 V  (fully depleted)
    OCV(1.0) ≈ 4.2 V  (fully charged)

Reference:
    Tremblay & Dessaint (2009). IEEE Trans. Veh. Technol., 58(8), 3961-3969.
    Panasonic NCR18650B datasheet (2013).
"""

import numpy as np


class NCABatteryF1a:
    """NCA battery cell model — voltage as a function of SOC and current only."""

    def __init__(self, params: dict):
        cell = params["cell"]
        ocv = params["ocv_coefficients"]

        self.capacity = cell["capacity"]["value"]               # Ah
        self.v_max = cell["voltage_max"]["value"]               # V
        self.v_min = cell["voltage_min"]["value"]               # V
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
        Terminal voltage [V].

        Args:
            soc:     State of charge (0-1)
            current: Current [A] (positive = discharge, negative = charge)

        Returns:
            Voltage clipped to [v_min, v_max]
        """
        soc = np.asarray(soc, dtype=float)
        current = np.asarray(current, dtype=float)
        v = self.ocv(soc) - current * self.r_internal
        return np.clip(v, self.v_min, self.v_max)

    def power(self, soc, current):
        """Electrical power [W] (positive = discharging, negative = charging)."""
        return self.terminal_voltage(soc, current) * np.asarray(current, dtype=float)

    def soc_derivative(self, soc, current):
        """dSOC/dt [1/s]. Integrate to get SOC trajectory."""
        return -np.asarray(current, dtype=float) / (self.capacity * 3600.0)
