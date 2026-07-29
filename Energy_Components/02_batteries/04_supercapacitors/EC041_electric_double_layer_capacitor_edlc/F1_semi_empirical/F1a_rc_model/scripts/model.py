"""
EC041 — Electric Double-Layer Capacitor (EDLC) — F1a RC Model

Simple linear RC model of an EDLC supercapacitor:

    V_capacitor    = Q / C
    V_terminal     = V_capacitor - I_term * ESR        (positive I = discharge)
    I_capacitor    = I_term + V_capacitor / R_leak     (leakage drains capacitor
                                                         even when external I=0)
    dQ/dt          = -I_capacitor

Stored energy: E = 0.5 * C * V_capacitor^2
SOC (state of charge) is defined as V_capacitor / V_max so that
SOC = 1 corresponds to a fully charged cell at the rated voltage.

Reference:
    Conway, B. E. (1999). Electrochemical Supercapacitors: Scientific
    Fundamentals and Technological Applications. Kluwer / Plenum.
"""

import numpy as np


class EDLCF1a:
    """EDLC supercapacitor cell model — linear RC with series ESR and leakage."""

    def __init__(self, params: dict):
        cell = params["cell"]
        self.C       = cell["capacitance"]["value"]         # F
        self.ESR     = cell["esr"]["value"]                  # Ohm
        self.v_max   = cell["v_max"]["value"]                # V
        self.v_min   = cell["v_min"]["value"]                # V
        self.R_leak  = cell["leakage_resistance"]["value"]   # Ohm

    # ---- algebraic outputs given (V_cap, I_terminal) ----

    def terminal_voltage(self, v_cap, current):
        """V_term = V_cap - I*ESR. Clipped to [v_min, v_max]."""
        v_cap = np.asarray(v_cap, dtype=float)
        current = np.asarray(current, dtype=float)
        v = v_cap - current * self.ESR
        return np.clip(v, self.v_min, self.v_max)

    def soc(self, v_cap):
        """SOC = V_cap / V_max, clipped to [0, 1]."""
        v_cap = np.asarray(v_cap, dtype=float)
        return np.clip(v_cap / self.v_max, 0.0, 1.0)

    def charge(self, v_cap):
        """Q = C * V_cap [Coulomb]."""
        return self.C * np.asarray(v_cap, dtype=float)

    def stored_energy(self, v_cap):
        """E = 0.5 * C * V_cap^2 [Joule]."""
        v_cap = np.asarray(v_cap, dtype=float)
        return 0.5 * self.C * v_cap * v_cap

    def power(self, v_cap, current):
        """Terminal electrical power [W]. Positive = discharging."""
        return self.terminal_voltage(v_cap, current) * np.asarray(current, dtype=float)

    def vcap_derivative(self, v_cap, current):
        """
        dV_cap/dt for the lumped RC cell:
            dV/dt = -I_cap / C
            I_cap = I_terminal + V_cap / R_leak
        Discharging (positive current) drives V_cap down.
        Leakage also drives V_cap down even when I_terminal = 0.
        """
        v_cap = np.asarray(v_cap, dtype=float)
        current = np.asarray(current, dtype=float)
        i_cap = current + v_cap / self.R_leak
        return -i_cap / self.C
