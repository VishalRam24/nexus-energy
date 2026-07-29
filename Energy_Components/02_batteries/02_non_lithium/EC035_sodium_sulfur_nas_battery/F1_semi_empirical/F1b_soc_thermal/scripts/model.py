"""
EC035 -- NaS Battery (Sodium-Sulfur) -- F1b SOC-Thermal Model

HIGH-TEMPERATURE CHEMISTRY: NaS cells require 300-350 degC to maintain molten
sodium and sulfur electrodes, and to keep beta-alumina solid electrolyte conductive.
OUTSIDE 300-350 degC: cell is NON-FUNCTIONAL (electrodes solidify).

Thermal model:
    R(T) = R_ref * exp(E_a / R_gas * (1/T - 1/T_op_ref))
    V    = OCV(SOC) - I * R(T)      [only valid when T in [T_op_min, T_op_max]]
    Q    = I^2 * R(T) + I * T * dOCV/dT
    C(T) = C_ref * (1 + alpha_c * (T - T_op_ref))

The `functional` flag in predict output is False when T < T_op_min or T > T_op_max.

References:
    Wen, Z. et al. (2008). Mater. Sci. Eng. B 154-155, 73-78.
    Sudworth, J. L. & Tilley, A. R. (1985). The Sodium Sulfur Battery. Chapman & Hall.
    NGK Insulators. NAS Battery Technical Brochure.
"""

import numpy as np


class NaSBatteryF1b:
    """NaS high-temperature battery cell model."""

    def __init__(self, params: dict):
        cell = params["cell"]
        ocv = params["ocv_coefficients"]
        therm = params["thermal"]

        self.capacity_ref = cell["capacity_ref"]["value"]
        self.v_max = cell["voltage_max"]["value"]
        self.v_min = cell["voltage_min"]["value"]
        self.R_ref = cell["R_ref"]["value"]

        self.ocv_coeff = np.array([ocv[f"a{i}"] for i in range(6)])

        self.T_op_ref = therm["T_op_ref"]["value"]   # 593.15 K = 320 degC
        self.T_op_min = therm["T_op_min"]["value"]   # 573.15 K = 300 degC
        self.T_op_max = therm["T_op_max"]["value"]   # 623.15 K = 350 degC
        self.E_a = therm["E_a"]["value"]
        self.alpha_c = therm["alpha_c"]["value"]
        self.dOCV_dT = therm["dOCV_dT"]["value"]
        self.R_gas = therm["R_gas"]["value"]

    def is_functional(self, temperature):
        """Returns True if cell is in operating temperature window (300-350 degC)."""
        temperature = np.asarray(temperature, dtype=float)
        return (temperature >= self.T_op_min) & (temperature <= self.T_op_max)

    def ocv(self, soc):
        """Open-circuit voltage as a function of SOC (0-1)."""
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([soc**i for i in range(6)], axis=-1)
        return np.dot(powers, self.ocv_coeff)

    def internal_resistance(self, temperature):
        """
        Temperature-dependent resistance via Arrhenius.
        Beta-alumina conductivity varies within the 300-350C operating window.
        """
        temperature = np.asarray(temperature, dtype=float)
        return self.R_ref * np.exp(
            self.E_a / self.R_gas * (1.0 / temperature - 1.0 / self.T_op_ref)
        )

    def effective_capacity(self, temperature):
        """Temperature-corrected capacity (within operating window)."""
        temperature = np.asarray(temperature, dtype=float)
        return self.capacity_ref * (1.0 + self.alpha_c * (temperature - self.T_op_ref))

    def terminal_voltage(self, soc, current, temperature):
        """
        Terminal voltage. Returns 0 V if cell outside 300-350 degC window
        (cell is non-functional: electrodes solidified or thermally unsafe).
        """
        soc = np.asarray(soc, dtype=float)
        current = np.asarray(current, dtype=float)
        temperature = np.asarray(temperature, dtype=float)
        functional = self.is_functional(temperature)
        R_T = self.internal_resistance(temperature)
        v = self.ocv(soc) - current * R_T
        v_clipped = np.clip(v, self.v_min, self.v_max)
        return np.where(functional, v_clipped, 0.0)

    def heat_generation(self, soc, current, temperature):
        """
        Total heat: Q = I^2*R(T) + I*T*dOCV/dT
        Zero if cell is outside operating window.
        """
        current = np.asarray(current, dtype=float)
        temperature = np.asarray(temperature, dtype=float)
        functional = self.is_functional(temperature)
        R_T = self.internal_resistance(temperature)
        q = current**2 * R_T + current * temperature * self.dOCV_dT
        return np.where(functional, q, 0.0)

    def power(self, soc, current, temperature):
        """Electrical power in W."""
        return self.terminal_voltage(soc, current, temperature) * np.asarray(current, dtype=float)

    def soc_derivative(self, current, temperature):
        """dSOC/dt in 1/s. Zero if cell is non-functional."""
        current = np.asarray(current, dtype=float)
        temperature = np.asarray(temperature, dtype=float)
        functional = self.is_functional(temperature)
        C_eff = self.effective_capacity(temperature)
        dsoc = -current / (C_eff * 3600.0)
        return np.where(functional, dsoc, 0.0)
