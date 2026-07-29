"""
EC041 -- EDLC Supercapacitor -- F1b RC-Thermal Model

Extends F1a by adding temperature dependence:
    ESR(T) = ESR_ref * exp(E_a / R_gas * (1/T - 1/T_ref))   [Arrhenius]
    C(T)   = C_ref * (1 + alpha_C * (T - T_ref))             [linear]
    V_term = V_cap - I * ESR(T)
    Q_gen  = I^2 * ESR(T)                                    [Joule heat; no entropic term for EDLC]

EDLC state variable is V_cap (capacitor voltage), not SOC per se.
SOC = V_cap / V_max is a convenience normalization.

Temperature range: -40 to 65 degC (EDLC works over very wide T range).

References:
    Conway, B. E. (1999). Electrochemical Supercapacitors. Kluwer/Plenum.
    Rafik et al. (2007). J. Power Sources 165, 928-934.
    Berrueta et al. (2019). IEEE Trans. Ind. Electron. 66(6), 4750-4759.
"""

import numpy as np


class EDLCF1b:
    """EDLC supercapacitor cell model with temperature-dependent ESR and capacitance."""

    def __init__(self, params: dict):
        cell = params["cell"]
        therm = params["thermal"]

        self.C_ref = cell["capacitance_ref"]["value"]        # F at T_ref
        self.ESR_ref = cell["esr_ref"]["value"]              # Ohm at T_ref
        self.v_max = cell["v_max"]["value"]                  # V
        self.v_min = cell["v_min"]["value"]                  # V
        self.R_leak = cell["leakage_resistance"]["value"]    # Ohm

        self.T_ref = therm["T_ref"]["value"]
        self.E_a = therm["E_a_esr"]["value"]
        self.alpha_C = therm["alpha_C"]["value"]
        self.R_gas = therm["R_gas"]["value"]

    def esr(self, temperature):
        """Temperature-dependent ESR via Arrhenius."""
        temperature = np.asarray(temperature, dtype=float)
        return self.ESR_ref * np.exp(
            self.E_a / self.R_gas * (1.0 / temperature - 1.0 / self.T_ref)
        )

    def capacitance(self, temperature):
        """Temperature-dependent capacitance (linear model)."""
        temperature = np.asarray(temperature, dtype=float)
        return self.C_ref * (1.0 + self.alpha_C * (temperature - self.T_ref))

    def soc(self, v_cap):
        """SOC = V_cap / V_max, clipped to [0, 1]."""
        v_cap = np.asarray(v_cap, dtype=float)
        return np.clip(v_cap / self.v_max, 0.0, 1.0)

    def charge(self, v_cap, temperature):
        """Q = C(T) * V_cap [Coulomb]."""
        return self.capacitance(temperature) * np.asarray(v_cap, dtype=float)

    def stored_energy(self, v_cap, temperature):
        """E = 0.5 * C(T) * V_cap^2 [Joule]."""
        v_cap = np.asarray(v_cap, dtype=float)
        return 0.5 * self.capacitance(temperature) * v_cap**2

    def terminal_voltage(self, v_cap, current, temperature):
        """V_term = V_cap - I * ESR(T), clipped to [v_min, v_max]."""
        v_cap = np.asarray(v_cap, dtype=float)
        current = np.asarray(current, dtype=float)
        v = v_cap - current * self.esr(temperature)
        return np.clip(v, self.v_min, self.v_max)

    def heat_generation(self, current, temperature):
        """
        Joule heat generation: Q = I^2 * ESR(T).
        EDLC has no electrochemical reactions, so no reversible (entropic) term.
        """
        current = np.asarray(current, dtype=float)
        return current**2 * self.esr(temperature)

    def power(self, v_cap, current, temperature):
        """Terminal power [W]. Positive = discharging."""
        return self.terminal_voltage(v_cap, current, temperature) * np.asarray(current, dtype=float)

    def vcap_derivative(self, v_cap, current, temperature):
        """
        dV_cap/dt for the lumped RC cell at given temperature:
            I_cap = I_terminal + V_cap / R_leak
            dV_cap/dt = -I_cap / C(T)
        """
        v_cap = np.asarray(v_cap, dtype=float)
        current = np.asarray(current, dtype=float)
        i_cap = current + v_cap / self.R_leak
        return -i_cap / self.capacitance(temperature)
