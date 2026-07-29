"""
EC026 -- Lithium-Air Battery (Li-O2) -- F1b SOC-Thermal Model

Extends F1a by adding temperature dependence via Arrhenius kinetics:
    R(T) = R_ref * exp(E_a / R_gas * (1/T - 1/T_ref))
    V    = OCV(SOC) - I * R(T)
    Q    = I^2 * R(T) + I * T * dOCV/dT     (irreversible + reversible heat)
    C(T) = C_ref * (1 + alpha_c * (T - T_ref))

Chemistry notes:
    Li-air (aprotic Li-O2) has the LARGEST negative dOCV/dT of common battery chemistries,
    approximately -0.50 mV/K. This is because:
    1. The cathode reaction involves gas-phase O2: 2Li + O2 -> Li2O2
    2. Gas adsorption/desorption at the electrode contributes large negative entropy
    3. ΔS = -n*F*(dOCV/dT) is large and positive (entropy decreases during discharge)

    The high activation energy (E_a = 35 kJ/mol) reflects the sluggish ORR/OER kinetics
    at the carbon air cathode with aprotic electrolyte (DMSO/TEGDME).

    The high R_ref (0.15 Ω) reflects the insulating Li2O2 film that passivates the cathode.

References:
    Laoire et al. (2010). J. Electrochem. Soc. 157(7), A821.
    Abraham & Jiang (1996). J. Electrochem. Soc. 143, 1.
    Viswanathan et al. (2011). J. Chem. Phys. 135, 214704.
    Lu et al. (2013). Nat. Chem. 5, 527.
"""

import numpy as np


class LiAirBatteryF1b:
    """Lithium-air (Li-O2) battery cell model -- voltage as a function of SOC, current, and temperature."""

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
        self.dOCV_dT = therm["dOCV_dT"]["value"]    # Large negative: -5.0e-4 V/K
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

        Li-air: dOCV/dT = -0.50 mV/K (most negative of common battery chemistries).
        During discharge (I > 0): reversible term is positive (exothermic entropic heat).
        Both terms are positive during discharge, giving significant total heating.
        This extra heating contributes to Li-air thermal management challenges.

        During charge (I < 0): reversible term is negative (endothermic), partially
        offsetting I^2*R Joule heating.
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
