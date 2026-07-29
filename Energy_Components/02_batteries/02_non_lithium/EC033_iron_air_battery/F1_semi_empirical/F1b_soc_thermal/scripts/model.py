"""
EC033 -- Iron-Air Battery -- F1b SOC-Thermal Model

Temperature-dependent model for iron-air secondary batteries.

    R(T) = R_ref * exp(E_a/R_gas * (1/T - 1/T_ref))   [Arrhenius]
    V    = OCV(SOC) - I * R(T)
    Q    = I^2 * R(T) + I * T * dOCV/dT                [irreversible + reversible heat]
    C(T) = C_ref * (1 + alpha_c * (T - T_ref))

Iron-Air specifics:
    - Anode: Fe + 2OH- -> Fe(OH)2 + 2e- (or Fe + 3OH- -> Fe(OH)3 + 3e-)
    - Cathode: O2 + H2O + 4e- -> 4OH-  (ORR)
    - E^0 ~ 1.28 V in alkaline KOH
    - Positive dOCV/dT (+0.0002 V/K): mild entropic exothermic effect on discharge
    - Slow electrode kinetics -> higher R_ref and E_a vs. Zn-air

References:
    Manohar et al. (2012). J. Electrochem. Soc. 159(8), A1209-A1214.
    Trocino et al. (2022). J. Power Sources 523, 230999.
    Form Energy (2022). Iron-air battery technology report.
"""

import numpy as np


class IronAirBatteryF1b:
    """Iron-air battery cell model -- voltage as a function of SOC, current, and temperature."""

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
        self.dOCV_dT  = float(th["dOCV_dT"]["value"])   # V/K; positive for Fe-air
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

        For Fe-air: dOCV/dT > 0, so during discharge (I > 0), reversible term
        is positive (exothermic contribution adds to Joule heating).
        """
        current     = np.asarray(current, dtype=float)
        temperature = np.asarray(temperature, dtype=float)
        R_T         = self.internal_resistance(temperature)
        return current**2 * R_T + current * temperature * self.dOCV_dT

    def power(self, soc, current, temperature):
        """Electrical power [W] (positive = discharging)."""
        return self.terminal_voltage(soc, current, temperature) * np.asarray(current, dtype=float)

    def soc_derivative(self, current, temperature):
        """dSOC/dt [1/s]."""
        current = np.asarray(current, dtype=float)
        C_eff   = self.effective_capacity(temperature)
        return -current / (C_eff * 3600.0)
