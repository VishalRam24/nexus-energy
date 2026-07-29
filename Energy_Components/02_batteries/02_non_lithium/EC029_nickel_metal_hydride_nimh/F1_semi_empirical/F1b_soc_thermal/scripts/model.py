"""
EC029 -- NiMH Battery (Nickel-Metal Hydride) -- F1b SOC-Thermal Model

Extends F1a by adding temperature dependence:
    R(T) = R_ref * exp(E_a / R_gas * (1/T - 1/T_ref))
    V    = OCV(SOC) - I * R(T)
    Q    = I^2 * R(T) + I * T * dOCV/dT
    C(T) = C_ref * (1 + alpha_c * (T - T_ref))

NiMH-specific characteristics:
    - Positive dOCV/dT (unlike Li-ion): charging is exothermic, reversible heat adds during charge
    - Wide operating range: -20 to 60 degC
    - Capacity drops significantly at low temperatures (~60% at -20C vs 25C)
    - Flat OCV plateau at ~1.2 V

References:
    Linden's Handbook of Batteries, 4th ed. (2011), ch. 31.
    Bernardi & Carpenter (1995). J. Electrochem. Soc. 142(8), 2631-2642.
    Khun et al. (2006). Electrochimica Acta 51, 2877-2887.
"""

import numpy as np


class NiMHBatteryF1b:
    """NiMH battery cell model -- voltage as a function of SOC, current, and temperature."""

    def __init__(self, params: dict):
        cell = params["cell"]
        ocv = params["ocv_coefficients"]
        therm = params["thermal"]

        self.capacity_ref = cell["capacity_ref"]["value"]
        self.v_max = cell["voltage_max"]["value"]
        self.v_min = cell["voltage_min"]["value"]
        self.R_ref = cell["R_ref"]["value"]

        self.ocv_coeff = np.array([ocv[f"a{i}"] for i in range(6)])

        self.T_ref = therm["T_ref"]["value"]
        self.E_a = therm["E_a"]["value"]
        self.alpha_c = therm["alpha_c"]["value"]
        self.dOCV_dT = therm["dOCV_dT"]["value"]   # positive for NiMH
        self.R_gas = therm["R_gas"]["value"]

    def ocv(self, soc):
        """Open-circuit voltage as a function of SOC (0-1)."""
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([soc**i for i in range(6)], axis=-1)
        return np.dot(powers, self.ocv_coeff)

    def internal_resistance(self, temperature):
        """Temperature-dependent internal resistance via Arrhenius."""
        temperature = np.asarray(temperature, dtype=float)
        return self.R_ref * np.exp(
            self.E_a / self.R_gas * (1.0 / temperature - 1.0 / self.T_ref)
        )

    def effective_capacity(self, temperature):
        """
        Temperature-corrected capacity.
        NiMH loses ~40% capacity at -20C; alpha_c captures this trend linearly.
        """
        temperature = np.asarray(temperature, dtype=float)
        cap = self.capacity_ref * (1.0 + self.alpha_c * (temperature - self.T_ref))
        return np.maximum(cap, 0.01)  # prevent negative capacity at extreme cold

    def terminal_voltage(self, soc, current, temperature):
        """Terminal voltage, clipped to [v_min, v_max]."""
        soc = np.asarray(soc, dtype=float)
        current = np.asarray(current, dtype=float)
        R_T = self.internal_resistance(temperature)
        v = self.ocv(soc) - current * R_T
        return np.clip(v, self.v_min, self.v_max)

    def heat_generation(self, soc, current, temperature):
        """
        Total heat: Q = I^2*R(T) + I*T*dOCV/dT
        NiMH dOCV/dT > 0: during discharge (I>0), reversible heat is positive (heat release).
        During charge (I<0), reversible term is negative (heat absorption), but Joule term
        typically dominates at practical rates.
        """
        current = np.asarray(current, dtype=float)
        temperature = np.asarray(temperature, dtype=float)
        R_T = self.internal_resistance(temperature)
        return current**2 * R_T + current * temperature * self.dOCV_dT

    def power(self, soc, current, temperature):
        """Electrical power in W."""
        return self.terminal_voltage(soc, current, temperature) * np.asarray(current, dtype=float)

    def soc_derivative(self, current, temperature):
        """dSOC/dt in 1/s, using temperature-corrected capacity."""
        current = np.asarray(current, dtype=float)
        C_eff = self.effective_capacity(temperature)
        return -current / (C_eff * 3600.0)
