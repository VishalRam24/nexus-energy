"""
EC023 -- LMO Battery (Lithium Manganese Oxide) -- F1b SOC-Thermal Model

Extends F1a by adding temperature dependence via Arrhenius kinetics:
    R(T) = R_ref * exp(E_a / R_gas * (1/T - 1/T_ref))
    V    = OCV(SOC) - I * R(T)
    Q    = I^2 * R(T) + I * T * dOCV/dT     (irreversible + reversible heat)
    C(T) = C_ref * (1 + alpha_c * (T - T_ref))

Chemistry notes:
    LMO (spinel LiMn2O4) has a characteristic flat OCV plateau near 4.0 V.
    dOCV/dT is mildly negative (-0.18 mV/K), less negative than NMC (-0.4 mV/K)
    because the Mn spinel structure has lower entropic penalty.
    Activation energy ~23 kJ/mol reflects higher manganese conductivity vs NMC.

References:
    Liaw et al. (2003). J. Power Sources, 119-121, 874-882.
    Jalkanen et al. (2015). Applied Energy 154, 160-172.
    Thomas & Newman (2003). J. Electrochem. Soc. 150, A176.
    Forgez et al. (2010). J. Power Sources 195, 2961-2968.
"""

import numpy as np


class LMOBatteryF1b:
    """LMO battery cell model -- voltage as a function of SOC, current, and temperature."""

    def __init__(self, params: dict):
        cell = params["cell"]
        ocv = params["ocv_coefficients"]
        therm = params["thermal"]

        self.capacity_ref = cell["capacity_ref"]["value"]       # Ah at T_ref
        self.v_max = cell["voltage_max"]["value"]               # V
        self.v_min = cell["voltage_min"]["value"]               # V
        self.R_ref = cell["R_ref"]["value"]                     # Ohm at T_ref

        # OCV polynomial coefficients (order 0 to 5)
        self.ocv_coeff = np.array([ocv[f"a{i}"] for i in range(6)])

        # Thermal parameters
        self.T_ref = therm["T_ref"]["value"]                    # K
        self.E_a = therm["E_a"]["value"]                        # J/mol
        self.alpha_c = therm["alpha_c"]["value"]                # 1/K
        self.dOCV_dT = therm["dOCV_dT"]["value"]                # V/K (negative for LMO)
        self.R_gas = therm["R_gas"]["value"]                    # J/(mol*K)

    def ocv(self, soc):
        """Open-circuit voltage as a function of SOC (0-1)."""
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([soc**i for i in range(6)], axis=-1)
        return np.dot(powers, self.ocv_coeff)

    def internal_resistance(self, temperature):
        """
        Temperature-dependent internal resistance via Arrhenius relation.

        Args:
            temperature: Temperature in K

        Returns:
            Internal resistance in Ohm
        """
        temperature = np.asarray(temperature, dtype=float)
        return self.R_ref * np.exp(
            self.E_a / self.R_gas * (1.0 / temperature - 1.0 / self.T_ref)
        )

    def effective_capacity(self, temperature):
        """
        Temperature-corrected capacity.

        Args:
            temperature: Temperature in K

        Returns:
            Effective capacity in Ah
        """
        temperature = np.asarray(temperature, dtype=float)
        return self.capacity_ref * (1.0 + self.alpha_c * (temperature - self.T_ref))

    def terminal_voltage(self, soc, current, temperature):
        """
        Terminal voltage.

        Args:
            soc: State of charge (0-1)
            current: Current in A (positive = discharge, negative = charge)
            temperature: Temperature in K

        Returns:
            Voltage in V (clipped to [v_min, v_max])
        """
        soc = np.asarray(soc, dtype=float)
        current = np.asarray(current, dtype=float)
        R_T = self.internal_resistance(temperature)
        v = self.ocv(soc) - current * R_T
        return np.clip(v, self.v_min, self.v_max)

    def heat_generation(self, soc, current, temperature):
        """
        Total heat generation rate.

        Q = I^2 * R(T)  +  I * T * dOCV/dT

        The first term is irreversible (Joule) heating, always positive.
        The second term is reversible (entropic) heating; for LMO dOCV/dT < 0,
        so discharge (I > 0) gives extra heating (negative entropy contribution).

        Args:
            soc: State of charge (0-1)
            current: Current in A (positive = discharge)
            temperature: Temperature in K

        Returns:
            Heat generation in W
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
