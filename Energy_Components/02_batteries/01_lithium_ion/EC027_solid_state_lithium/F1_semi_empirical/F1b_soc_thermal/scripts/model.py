"""
EC027 -- Solid-State Lithium Battery -- F1b SOC-Thermal Model

Extends F1a by adding temperature dependence via Arrhenius kinetics:
    R(T) = R_ref * exp(E_a / R_gas * (1/T - 1/T_ref))
    V    = OCV(SOC) - I * R(T)
    Q    = I^2 * R(T) + I * T * dOCV/dT     (irreversible + reversible heat)
    C(T) = C_ref * (1 + alpha_c * (T - T_ref))

Solid-state characteristics:
    - Sulfide solid electrolyte (Li6PS5Cl, LGPS-class) with E_a ~ 35 kJ/mol
    - Ionic conductivity strongly temperature-dependent (Arrhenius)
    - Operating range: 0-60 degC; warmer operation beneficial for SSE conductivity
    - Higher R_ref than liquid-electrolyte cells due to solid-solid interfacial resistance

References:
    Janek & Zeier (2016). Nat. Energy 1, 16141.
    Takada (2013). Acta Materialia 61, 759-770.
    Thomas & Newman (2003). J. Electrochem. Soc. 150(2), A176-A192.
"""

import numpy as np


class SolidStateLiBatteryF1b:
    """Solid-state Li battery cell model -- voltage as function of SOC, current, and temperature."""

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
        self.dOCV_dT = therm["dOCV_dT"]["value"]
        self.R_gas = therm["R_gas"]["value"]

    def ocv(self, soc):
        """Open-circuit voltage as a function of SOC (0-1)."""
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([soc**i for i in range(6)], axis=-1)
        return np.dot(powers, self.ocv_coeff)

    def internal_resistance(self, temperature):
        """Temperature-dependent resistance via Arrhenius (SSE ionic conductivity)."""
        temperature = np.asarray(temperature, dtype=float)
        return self.R_ref * np.exp(
            self.E_a / self.R_gas * (1.0 / temperature - 1.0 / self.T_ref)
        )

    def effective_capacity(self, temperature):
        """Temperature-corrected capacity."""
        temperature = np.asarray(temperature, dtype=float)
        return self.capacity_ref * (1.0 + self.alpha_c * (temperature - self.T_ref))

    def terminal_voltage(self, soc, current, temperature):
        """Terminal voltage, clipped to [v_min, v_max]."""
        soc = np.asarray(soc, dtype=float)
        current = np.asarray(current, dtype=float)
        R_T = self.internal_resistance(temperature)
        v = self.ocv(soc) - current * R_T
        return np.clip(v, self.v_min, self.v_max)

    def heat_generation(self, soc, current, temperature):
        """Total heat generation: Q = I^2*R(T) + I*T*dOCV/dT."""
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
