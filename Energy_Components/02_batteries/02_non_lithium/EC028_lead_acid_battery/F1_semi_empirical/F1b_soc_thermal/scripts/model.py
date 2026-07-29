"""
EC028 -- Lead-Acid Battery -- F1b SOC-Thermal Model

Extends F1a by adding temperature dependence via Arrhenius kinetics:
    R(T) = R_ref * exp(E_a / R_gas * (1/T - 1/T_ref))
    V    = OCV(SOC) - I * R(T)
    Q    = I^2 * R(T) + I * T * dOCV/dT     (irreversible + reversible heat)
    C(T) = C_ref * (1 + alpha_c * (T - T_ref))

Lead-acid characteristics:
    - Lower E_a (~15 kJ/mol) than Li-ion
    - Strong capacity-temperature dependence (alpha_c ~0.01/K)
    - Peukert exponent n=1.2 for rate-dependent capacity
    - 6-cell series (12V nominal)

References:
    Copetti et al. (1993). Progress in Photovoltaics, 1(4), 283-292.
    Manwell & McGowan (1993). Solar Energy, 50(5), 399-405.
    Bode (1977). Lead-Acid Batteries, Wiley.
    Schiffer et al. (2007). J. Power Sources, 168, 66-78.
"""

import numpy as np


class LeadAcidF1b:
    """Lead-acid battery model -- voltage as a function of SOC, current, and temperature."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V_nom = u["V_nom"]["value"]
        self.capacity_ref = u["capacity_ref"]["value"]          # Ah at T_ref
        self.V_min = u["voltage_min"]["value"]
        self.V_max = u["voltage_max"]["value"]
        self.R_ref = u["R_ref"]["value"]
        self.peukert_n = u["peukert_n"]["value"]

        c = u["ocv_coeffs"]
        self.a0 = c["a0"]["value"]
        self.a1 = c["a1"]["value"]
        self.a2 = c["a2"]["value"]
        self.a3 = c["a3"]["value"]

        therm = params["thermal"]
        self.T_ref = therm["T_ref"]["value"]
        self.E_a = therm["E_a"]["value"]
        self.alpha_c = therm["alpha_c"]["value"]
        self.dOCV_dT = therm["dOCV_dT"]["value"]
        self.R_gas = therm["R_gas"]["value"]

    def ocv(self, soc):
        """Open-circuit voltage [V] as a cubic polynomial of SOC."""
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        return self.a0 + self.a1 * soc + self.a2 * soc**2 + self.a3 * soc**3

    def internal_resistance(self, temperature):
        """Temperature-dependent internal resistance via Arrhenius relation [Ohm]."""
        temperature = np.asarray(temperature, dtype=float)
        return self.R_ref * np.exp(
            self.E_a / self.R_gas * (1.0 / temperature - 1.0 / self.T_ref)
        )

    def effective_capacity(self, temperature):
        """Temperature-corrected capacity [Ah]."""
        temperature = np.asarray(temperature, dtype=float)
        return self.capacity_ref * (1.0 + self.alpha_c * (temperature - self.T_ref))

    def terminal_voltage(self, soc, current, temperature):
        """Terminal voltage [V]. current > 0 = discharge."""
        soc = np.asarray(soc, dtype=float)
        current = np.asarray(current, dtype=float)
        R_T = self.internal_resistance(temperature)
        v = self.ocv(soc) - current * R_T
        return np.clip(v, self.V_min, self.V_max)

    def heat_generation(self, soc, current, temperature):
        """Total heat generation [W]: Q = I^2*R(T) + I*T*dOCV/dT."""
        current = np.asarray(current, dtype=float)
        temperature = np.asarray(temperature, dtype=float)
        R_T = self.internal_resistance(temperature)
        # dOCV_dT is for the full battery (all 6 cells combined)
        return current**2 * R_T + current * temperature * self.dOCV_dT

    def power(self, soc, current, temperature):
        """Terminal power [W]. Positive = discharge."""
        return self.terminal_voltage(soc, current, temperature) * np.asarray(current, dtype=float)

    def soc_derivative(self, current, temperature):
        """dSOC/dt [1/s], using temperature-corrected capacity."""
        current = np.asarray(current, dtype=float)
        C_eff = self.effective_capacity(temperature)
        return -current / (C_eff * 3600.0)
