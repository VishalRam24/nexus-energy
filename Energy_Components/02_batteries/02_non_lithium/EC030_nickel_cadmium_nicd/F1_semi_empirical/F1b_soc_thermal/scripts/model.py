"""
EC030 -- Nickel-Cadmium Battery (NiCd) -- F1b SOC-Thermal Model

Extends F1a by adding temperature dependence via Arrhenius kinetics:
    R(T) = R_ref * exp(E_a / R_gas * (1/T - 1/T_ref))
    V    = OCV(SOC) - I * R(T)
    Q    = I^2 * R(T) + I * T * dOCV/dT     (irreversible + reversible heat)
    C(T) = C_ref * (1 + alpha_c * (T - T_ref))

Chemistry notes:
    NiCd has dOCV/dT ≈ -0.60 mV/K (large negative, comparable to Li-air).
    The NiOOH cathode reaction: NiOOH + H2O + e- → Ni(OH)2 + OH-
    has large negative reaction entropy due to ordering of water molecules.

    NiCd has very low internal resistance (0.010 Ω for 10Ah cell) due to:
    - Sintered plate construction with large active surface area
    - Aqueous KOH electrolyte with high ionic conductivity
    - Low E_a (20 kJ/mol) since KOH conductivity is less temperature-sensitive than organic electrolytes

    The memory effect (not modeled here) can reduce effective capacity after repeated
    shallow cycling. This F1b model assumes no memory effect.

References:
    Berndt (2003). Maintenance-Free Batteries. Wiley.
    Shepherd (1965). J. Electrochem. Soc. 112, 657.
    Thomas & Newman (2003). J. Electrochem. Soc. 150, A176.
"""

import numpy as np


class NiCdBatteryF1b:
    """Nickel-Cadmium battery cell model -- voltage as a function of SOC, current, and temperature."""

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
        self.dOCV_dT = therm["dOCV_dT"]["value"]    # Large negative: -6.0e-4 V/K
        self.R_gas = therm["R_gas"]["value"]

    def ocv(self, soc):
        """Open-circuit voltage as a function of SOC (0-1)."""
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([soc**i for i in range(6)], axis=-1)
        return np.dot(powers, self.ocv_coeff)

    def internal_resistance(self, temperature):
        """Temperature-dependent internal resistance via Arrhenius relation."""
        temperature = np.asarray(temperature, dtype=float)
        return self.R_ref * np.exp(
            self.E_a / self.R_gas * (1.0 / temperature - 1.0 / self.T_ref)
        )

    def effective_capacity(self, temperature):
        """Temperature-corrected capacity."""
        temperature = np.asarray(temperature, dtype=float)
        return self.capacity_ref * (1.0 + self.alpha_c * (temperature - self.T_ref))

    def terminal_voltage(self, soc, current, temperature):
        """Terminal voltage (clipped to [v_min, v_max])."""
        soc = np.asarray(soc, dtype=float)
        current = np.asarray(current, dtype=float)
        R_T = self.internal_resistance(temperature)
        v = self.ocv(soc) - current * R_T
        return np.clip(v, self.v_min, self.v_max)

    def heat_generation(self, soc, current, temperature):
        """
        Total heat generation rate.

        Q = I^2 * R(T)  +  I * T * dOCV/dT

        NiCd: dOCV/dT = -0.60 mV/K (large negative, comparable to Li-air).
        During discharge (I > 0): both Joule and entropic heat are positive.
        NiCd cells generate significant heat during high-rate discharge.
        """
        current = np.asarray(current, dtype=float)
        temperature = np.asarray(temperature, dtype=float)
        R_T = self.internal_resistance(temperature)
        q_irreversible = current**2 * R_T
        q_reversible = current * temperature * self.dOCV_dT
        return q_irreversible + q_reversible

    def power(self, soc, current, temperature):
        """Electrical power in W (positive = discharging, negative = charging)."""
        return self.terminal_voltage(soc, current, temperature) * np.asarray(current, dtype=float)

    def soc_derivative(self, current, temperature):
        """dSOC/dt in 1/s, using temperature-corrected capacity."""
        current = np.asarray(current, dtype=float)
        C_eff = self.effective_capacity(temperature)
        return -current / (C_eff * 3600.0)
