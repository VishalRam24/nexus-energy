"""
EC025 -- Lithium-Sulfur Battery (Li-S) -- F1b SOC-Thermal Model

Extends F1a by adding temperature dependence via Arrhenius kinetics:
    R(T) = R_ref * exp(E_a / R_gas * (1/T - 1/T_ref))
    V    = OCV(SOC) - I * R(T)
    Q    = I^2 * R(T) + I * T * dOCV/dT     (irreversible + reversible heat)
    C(T) = C_ref * (1 + alpha_c * (T - T_ref))

CRITICAL CHEMISTRY NOTE — dOCV/dT is POSITIVE for Li-S:
    Unlike Li-ion cathodes (NMC, LFP, LMO) which have dOCV/dT < 0,
    Li-S has dOCV/dT ≈ +0.35 mV/K. This is because the overall reaction
    Li + 0.5S → 0.5Li2S is entropy-producing (positive ΔS), so:
      - During discharge (I > 0): reversible heat term = I * T * dOCV/dT < 0
        (endothermic — the cell absorbs heat, partially cooling itself)
      - This partially offsets I²R Joule heating
    At low currents, the reversible endothermic effect can dominate, giving
    net Q < 0 (net cooling during discharge), which is physically correct for Li-S.

References:
    Wild et al. (2015). Energy Environ. Sci. 8, 3477.
    Mikhaylik & Akridge (2004). J. Electrochem. Soc. 151, A1969.
    Kumaresan et al. (2008). J. Electrochem. Soc. 155(6), A576.
    Cuisinier et al. (2014). J. Phys. Chem. Lett. 5, 3227.
"""

import numpy as np


class LiSBatteryF1b:
    """Lithium-sulfur battery cell model -- voltage as a function of SOC, current, and temperature."""

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
        self.dOCV_dT = therm["dOCV_dT"]["value"]    # POSITIVE for Li-S: +3.5e-4 V/K
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
        """
        Temperature-corrected capacity.
        Li-S capacity increases strongly with temperature (alpha_c = 0.008 /K)
        due to enhanced polysulfide dissolution kinetics.
        """
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

        For Li-S: dOCV/dT = +0.35 mV/K (POSITIVE).
        During discharge (I > 0): reversible term is NEGATIVE (endothermic).
        At low currents, Q_total may be negative (net cooling).
        At high currents, I^2*R dominates and Q > 0.

        This is a well-established thermodynamic property of Li-S cells:
        ref. Kumaresan et al. (2008), J. Electrochem. Soc. 155(6), A576.
        """
        current = np.asarray(current, dtype=float)
        temperature = np.asarray(temperature, dtype=float)
        R_T = self.internal_resistance(temperature)
        q_irreversible = current**2 * R_T
        q_reversible = current * temperature * self.dOCV_dT  # NOTE: positive dOCV/dT, negative during discharge
        return q_irreversible + q_reversible

    def power(self, soc, current, temperature):
        """Electrical power in W (positive = discharging, negative = charging)."""
        return self.terminal_voltage(soc, current, temperature) * np.asarray(current, dtype=float)

    def soc_derivative(self, current, temperature):
        """dSOC/dt in 1/s, using temperature-corrected capacity."""
        current = np.asarray(current, dtype=float)
        C_eff = self.effective_capacity(temperature)
        return -current / (C_eff * 3600.0)
