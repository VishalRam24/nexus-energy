"""
EC032 -- Zinc-Air Battery -- F1b SOC-Thermal Model

Temperature-dependent model for zinc-air electrochemical cells.

    R(T) = R_ref * exp(E_a/R_gas * (1/T - 1/T_ref))   [Arrhenius]
    V    = OCV(SOC) - I * R(T)
    Q    = I^2 * R(T) + I * T * dOCV/dT                [irreversible + reversible heat]
    C(T) = C_ref * (1 + alpha_c * (T - T_ref))

Zinc-Air specifics:
    - Cathode: O2 + H2O + 2e- -> 2OH-  (ORR at air electrode)
    - Anode:   Zn + 2OH- -> Zn(OH)2 -> ZnO + H2O + 2e-
    - E^0 = 1.65 V (OCV at full charge, ambient)
    - dOCV/dT < 0: entropy decreases on discharge (Parker 2017)
    - Higher activation energy E_a (alkaline electrolyte, O2 transport)

References:
    Fu et al. (2010). J. Electrochem. Soc. 157(1), A50-A56.
    Lee et al. (2011). Int. J. Hydrogen Energy 36(14), 8430-8440.
    Parker et al. (2017). Science 356(6345), 415-418.
    Chotard et al. (2014). J. Electrochem. Soc.
"""

import numpy as np


class ZincAirBatteryF1b:
    """Zinc-air battery cell model -- voltage as a function of SOC, current, and temperature."""

    def __init__(self, params: dict):
        cell = params["cell"]
        ocv  = params["ocv_coefficients"]
        th   = params["thermal"]

        self.capacity_ref = float(cell["capacity_ref"]["value"])
        self.v_max        = float(cell["voltage_max"]["value"])
        self.v_min        = float(cell["voltage_min"]["value"])
        self.R_ref        = float(cell["R_ref"]["value"])

        self.ocv_coeff    = np.array([ocv[f"a{i}"] for i in range(6)])

        self.T_ref        = float(th["T_ref"]["value"])
        self.E_a          = float(th["E_a"]["value"])
        self.alpha_c      = float(th["alpha_c"]["value"])
        self.dOCV_dT      = float(th["dOCV_dT"]["value"])   # V/K; negative for Zn-air
        self.R_gas        = float(th["R_gas"]["value"])

    def ocv(self, soc):
        """Open-circuit voltage as a function of SOC (0-1)."""
        soc   = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([soc**i for i in range(6)], axis=-1)
        return np.dot(powers, self.ocv_coeff)

    def internal_resistance(self, temperature):
        """Temperature-dependent internal resistance via Arrhenius [Ohm]."""
        temperature = np.asarray(temperature, dtype=float)
        return self.R_ref * np.exp(
            self.E_a / self.R_gas * (1.0 / temperature - 1.0 / self.T_ref)
        )

    def effective_capacity(self, temperature):
        """Temperature-corrected capacity [Ah]."""
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
        """
        Total heat generation [W]: Q = I^2*R(T) + I*T*dOCV/dT.

        For Zn-air: dOCV/dT < 0, so during discharge (I > 0), reversible term
        is negative (endothermic contribution, slight self-cooling effect).
        Irreversible Joule heating dominates at normal currents.
        """
        current     = np.asarray(current, dtype=float)
        temperature = np.asarray(temperature, dtype=float)
        R_T         = self.internal_resistance(temperature)
        q_irrev     = current**2 * R_T
        q_rev       = current * temperature * self.dOCV_dT
        return q_irrev + q_rev

    def power(self, soc, current, temperature):
        """Electrical power [W] (positive = discharging)."""
        return self.terminal_voltage(soc, current, temperature) * np.asarray(current, dtype=float)

    def soc_derivative(self, current, temperature):
        """dSOC/dt [1/s], using temperature-corrected capacity."""
        current = np.asarray(current, dtype=float)
        C_eff   = self.effective_capacity(temperature)
        return -current / (C_eff * 3600.0)
