"""
EC042 -- Pseudocapacitor -- F1b RC-Thermal Model

Extends F1a by adding temperature dependence:
    ESR(T) = ESR_ref * exp(E_a / R_gas * (1/T - 1/T_ref))   [Arrhenius; higher E_a than EDLC]
    C(T)   = C_ref * (1 + alpha_C * (T - T_ref))             [linear; slightly higher alpha than EDLC]
    V_term = V_cap - I * ESR(T)
    Q_gen  = I^2 * ESR(T) + I * V_cap * dOCV_dT_specific     [Joule + small faradaic entropic term]

Key distinction from EDLC (EC041):
    - Higher ESR (faradaic charge-transfer resistance at RuO2 surface)
    - Higher E_a for ESR (12 kJ/mol vs 8 kJ/mol for EDLC)
    - Entropic heat term from proton-insertion reactions (absent in EDLC)
    - Higher leakage current (faradaic side reactions)
    - Narrower T range (aqueous electrolyte)

Representative chemistry: RuO2 in H2SO4
    RuO2 + H+ + e- <-> RuO(OH)   (proton-coupled electron transfer)
    Voltage window: 0-1 V in 1 M H2SO4

Temperature range: -30 to 60 degC (aqueous electrolyte constraints).

References:
    Conway, B. E. (1999). Electrochemical Supercapacitors. Kluwer/Plenum.
    Trasatti, S., Buzzanca, G. (1971). J. Electroanal. Chem. 29, A1-A5.
    Zheng, J. P. et al. (1995). J. Electrochem. Soc. 142, 2699-2703.
    Sugimoto, W. et al. (2006). Electrochim. Acta 52, 1742-1748.
    Simon, P., Gogotsi, Y. (2008). Nature Materials 7, 845-854.
"""

import numpy as np


class PseudocapacitorF1b:
    """Pseudocapacitor cell model with temperature-dependent ESR, capacitance, and entropic heat."""

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
        self.dOCV_dT = therm["dOCV_dT_specific"]["value"]   # V/K (entropic coefficient)
        self.R_gas = therm["R_gas"]["value"]

    def esr(self, temperature):
        """Temperature-dependent ESR via Arrhenius (includes charge-transfer contribution)."""
        temperature = np.asarray(temperature, dtype=float)
        return self.ESR_ref * np.exp(
            self.E_a / self.R_gas * (1.0 / temperature - 1.0 / self.T_ref)
        )

    def capacitance(self, temperature):
        """Temperature-dependent pseudocapacitance (linear model)."""
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

    def leakage_current(self, v_cap):
        """Self-discharge leakage current [A]. I_leak = V_cap / R_leak."""
        v_cap = np.asarray(v_cap, dtype=float)
        return v_cap / self.R_leak

    def heat_generation(self, v_cap, current, temperature):
        """
        Heat generation [W].
        Q = I^2 * ESR(T)  [Joule, dominant]
          + I * V_cap * |dOCV_dT|  [small faradaic entropic contribution from proton insertion]

        The entropic term represents partial battery-like behavior of pseudocapacitors.
        For EDLC there is no electrochemistry, so Q = I^2*ESR only.
        Here, proton-coupled electron transfer (RuO2) contributes a small reversible heat.
        Sign: dOCV_dT < 0 for proton-insertion, so discharge (I>0) releases additional heat.
        """
        v_cap = np.asarray(v_cap, dtype=float)
        current = np.asarray(current, dtype=float)
        q_joule = current**2 * self.esr(temperature)
        # Entropic contribution: Q_rev = I * V_cap * |dOCV_dT| for proton insertion
        # dOCV_dT < 0, so at discharge (I>0): q_entropic > 0 (heat released)
        q_entropic = current * v_cap * (-self.dOCV_dT)
        return q_joule + q_entropic

    def power(self, v_cap, current, temperature):
        """Terminal power [W]. Positive = discharging."""
        return self.terminal_voltage(v_cap, current, temperature) * np.asarray(current, dtype=float)

    def vcap_derivative(self, v_cap, current, temperature):
        """
        dV_cap/dt for the lumped RC pseudocapacitor at given temperature:
            I_cap = I_terminal + I_leak
            dV_cap/dt = -I_cap / C(T)
        where I_leak = V_cap / R_leak (self-discharge)
        """
        v_cap = np.asarray(v_cap, dtype=float)
        current = np.asarray(current, dtype=float)
        i_total = current + self.leakage_current(v_cap)
        return -i_total / self.capacitance(temperature)
