"""
EC034 -- Aluminum-Ion Battery -- F1b SOC-Thermal Model

Temperature-dependent model for aluminum-ion cells using ionic liquid electrolyte.

    R(T) = R_ref * exp(E_a/R_gas * (1/T - 1/T_ref))   [Arrhenius]
    V    = OCV(SOC) - I * R(T)
    Q    = I^2 * R(T) + I * T * dOCV/dT                [irreversible + reversible heat]
    C(T) = C_ref * (1 + alpha_c * (T - T_ref))

Aluminum-Ion specifics:
    - Cathode: AlCl4- intercalation into graphite layers
    - Anode:   Al dissolution/deposition: Al - 3e- -> Al^3+
    - Electrolyte: AlCl3:EMImCl ionic liquid
    - Nominal voltage ~2.0 V, high rate capability
    - Capacity strongly T-dependent (ionic liquid viscosity)
    - dOCV/dT < 0 (slight entropy decrease on Al intercalation)

References:
    Lin et al. (2015). Nature 520, 324-328.
    Pang et al. (2019). Joule 3(1), 136-148.
    Guo et al. (2020). Energy Storage Mater. 28, 240-248.
"""

import numpy as np


class AluminumIonBatteryF1b:
    """Aluminum-ion battery cell model -- voltage as a function of SOC, current, and temperature."""

    def __init__(self, params: dict):
        cell = params["cell"]
        ocv  = params["ocv_coefficients"]
        th   = params["thermal"]

        self.capacity_ref = float(cell["capacity_ref"]["value"])
        self.v_max        = float(cell["voltage_max"]["value"])
        self.v_min        = float(cell["voltage_min"]["value"])
        self.R_ref        = float(cell["R_ref"]["value"])

        self.ocv_coeff = np.array([ocv[f"a{i}"] for i in range(6)])

        self.T_ref    = float(th["T_ref"]["value"])
        self.E_a      = float(th["E_a"]["value"])
        self.alpha_c  = float(th["alpha_c"]["value"])
        self.dOCV_dT  = float(th["dOCV_dT"]["value"])
        self.R_gas    = float(th["R_gas"]["value"])

    def ocv(self, soc):
        """Open-circuit voltage as a function of SOC (0-1)."""
        soc    = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([soc**i for i in range(6)], axis=-1)
        return np.dot(powers, self.ocv_coeff)

    def internal_resistance(self, temperature):
        """Temperature-dependent internal resistance via Arrhenius [Ohm]."""
        temperature = np.asarray(temperature, dtype=float)
        return self.R_ref * np.exp(
            self.E_a / self.R_gas * (1.0 / temperature - 1.0 / self.T_ref)
        )

    def effective_capacity(self, temperature):
        """Temperature-corrected capacity [Ah]. Al-ion is more T-sensitive (ionic liquid viscosity)."""
        temperature = np.asarray(temperature, dtype=float)
        return self.capacity_ref * (1.0 + self.alpha_c * (temperature - self.T_ref))

    def terminal_voltage(self, soc, current, temperature):
        """Terminal voltage [V]. current: positive=discharge, negative=charge."""
        soc     = np.asarray(soc, dtype=float)
        current = np.asarray(current, dtype=float)
        R_T     = self.internal_resistance(temperature)
        v       = self.ocv(soc) - current * R_T
        return np.clip(v, self.v_min, self.v_max)

    def heat_generation(self, soc, current, temperature):
        """Total heat generation [W]: Q = I^2*R(T) + I*T*dOCV/dT."""
        current     = np.asarray(current, dtype=float)
        temperature = np.asarray(temperature, dtype=float)
        R_T         = self.internal_resistance(temperature)
        return current**2 * R_T + current * temperature * self.dOCV_dT

    def power(self, soc, current, temperature):
        """Electrical power [W]."""
        return self.terminal_voltage(soc, current, temperature) * np.asarray(current, dtype=float)

    def soc_derivative(self, current, temperature):
        """dSOC/dt [1/s]."""
        current = np.asarray(current, dtype=float)
        C_eff   = self.effective_capacity(temperature)
        return -current / (C_eff * 3600.0)
