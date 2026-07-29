"""
EC031 — Sodium-Ion Battery — F1a SOC-Only Model

Semi-empirical voltage model adapted from the Tremblay & Dessaint framework for Na-ion chemistry:
    V_terminal = OCV(SOC) - I * R_internal

OCV(SOC) 5th-order polynomial calibrated to Na-ion characteristics:
    OCV = a0 + a1*SOC + a2*SOC^2 + a3*SOC^3 + a4*SOC^4 + a5*SOC^5

Coefficients give:
    OCV(0.0) ≈ 2.2 V  (fully depleted)
    OCV(1.0) ≈ 3.8 V  (fully charged)

Key Na-ion vs Li-ion differences captured:
    - Lower average voltage (~3.1 V nominal vs ~3.6 V NMC/NCA)
    - Higher internal resistance (0.050 Ohm vs 0.030-0.045 Ohm for Li-ion)
    - Wider operating voltage window due to hard-carbon anode

Reference:
    Tremblay & Dessaint (2009). IEEE Trans. Veh. Technol., 58(8), 3961-3969.
    CATL Na-ion battery press release (2021); Slater et al. (2013), Adv. Funct. Mater.
"""

import numpy as np


class NaIonBatteryF1a:
    """Sodium-ion battery cell model — voltage as a function of SOC and current."""

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
