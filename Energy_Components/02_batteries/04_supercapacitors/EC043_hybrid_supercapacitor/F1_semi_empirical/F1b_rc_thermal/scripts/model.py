"""
EC043 -- Hybrid Supercapacitor -- F1b RC-Thermal Model

Temperature-dependent RC model for hybrid supercapacitors (lithium-ion capacitors,
asymmetric pseudocapacitors). Combines EDLC-type double-layer capacitance with
battery-type intercalation electrode.

    ESR(T)  = ESR_ref * exp(E_a/R_gas * (1/T - 1/T_ref))   [Arrhenius]
    C(T)    = C_ref * (1 + alpha_C * (T - T_ref))           [linear]
    V_term  = V_cap - I * ESR(T)
    Q_gen   = I^2 * ESR(T)                                  [Joule heat only; no significant entropic term]

Hybrid SC differences from EDLC:
    - Higher energy density (battery-type pre-intercalation anode)
    - Asymmetric voltage window (1.8 V to 3.8 V for LIC)
    - Lower ESR than EDLC, lower E_a for ESR (better thermal behavior)
    - Mild capacitance T-dependence

References:
    Zhang & Zhao (2009). Chem. Soc. Rev. 38(9), 2520-2531.
    Naoi et al. (2012). Energy Environ. Sci. 5(11), 9363-9373.
    Berrueta et al. (2019). IEEE Trans. Ind. Electron. 66(6), 4750-4759.
"""

import numpy as np


class HybridSupercapacitorF1b:
    """Hybrid supercapacitor with temperature-dependent ESR and capacitance."""

    def __init__(self, params: dict):
        cell = params["cell"]
        th   = params["thermal"]

        self.C_ref    = float(cell["capacitance_ref"]["value"])     # F at T_ref
        self.ESR_ref  = float(cell["esr_ref"]["value"])             # Ohm at T_ref
        self.v_max    = float(cell["v_max"]["value"])               # V
        self.v_min    = float(cell["v_min"]["value"])               # V
        self.R_leak   = float(cell["leakage_resistance"]["value"])  # Ohm

        self.T_ref    = float(th["T_ref"]["value"])
        self.E_a      = float(th["E_a_esr"]["value"])
        self.alpha_C  = float(th["alpha_C"]["value"])
        self.R_gas    = float(th["R_gas"]["value"])

    def esr(self, temperature):
        """Temperature-dependent ESR [Ohm] via Arrhenius."""
        temperature = np.asarray(temperature, dtype=float)
        return self.ESR_ref * np.exp(
            self.E_a / self.R_gas * (1.0 / temperature - 1.0 / self.T_ref)
        )

    def capacitance(self, temperature):
        """Temperature-dependent capacitance [F] (linear model)."""
        temperature = np.asarray(temperature, dtype=float)
        return self.C_ref * (1.0 + self.alpha_C * (temperature - self.T_ref))

    def soc(self, v_cap):
        """
        SOC = (V_cap^2 - V_min^2) / (V_max^2 - V_min^2).
        Accounts for energy stored proportional to V^2 (unlike battery SOC).
        """
        v_cap = np.asarray(v_cap, dtype=float)
        v_safe = np.clip(v_cap, self.v_min, self.v_max)
        return (v_safe**2 - self.v_min**2) / (self.v_max**2 - self.v_min**2)

    def stored_energy(self, v_cap, temperature):
        """Stored energy [J] = 0.5 * C(T) * V_cap^2."""
        v_cap = np.asarray(v_cap, dtype=float)
        return 0.5 * self.capacitance(temperature) * v_cap**2

    def terminal_voltage(self, v_cap, current, temperature):
        """V_term = V_cap - I * ESR(T), clipped to [v_min, v_max]."""
        v_cap   = np.asarray(v_cap, dtype=float)
        current = np.asarray(current, dtype=float)
        v       = v_cap - current * self.esr(temperature)
        return np.clip(v, self.v_min, self.v_max)

    def heat_generation(self, current, temperature):
        """
        Joule heat generation [W]: Q = I^2 * ESR(T).
        Hybrid SC has negligible entropic contribution (mixed storage mechanism).
        """
        current = np.asarray(current, dtype=float)
        return current**2 * self.esr(temperature)

    def power(self, v_cap, current, temperature):
        """Terminal power [W]. Positive = discharging."""
        return self.terminal_voltage(v_cap, current, temperature) * np.asarray(current, dtype=float)

    def vcap_derivative(self, v_cap, current, temperature):
        """
        dV_cap/dt [V/s] for lumped RC:
            I_cap = I + V_cap / R_leak    (leakage discharges the cap)
            dV_cap/dt = -I_cap / C(T)
        """
        v_cap   = np.asarray(v_cap, dtype=float)
        current = np.asarray(current, dtype=float)
        I_cap   = current + v_cap / self.R_leak
        return -I_cap / self.capacitance(temperature)
