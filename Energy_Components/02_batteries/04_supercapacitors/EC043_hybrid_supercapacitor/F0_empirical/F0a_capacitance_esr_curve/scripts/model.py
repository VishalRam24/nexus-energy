"""F0a empirical capacitance/ESR lookup for Hybrid Supercapacitor (EC043).

F0 = simplest fidelity: a linear-capacitor curve V = Q/C with an ESR voltage
drop, giving stored energy 0.5*C*V^2 and a DC round-trip efficiency from the
ESR loss. No dynamics, no ODE -- a closed-form algebraic lookup.

Data source (reused from EC043 F1a):
EC043 F1a (C=200 F, ESR=10 mOhm, V_max=3.8 V, Q_max=760 C); Conway (1999); Li-ion capacitor datasheets. DC round-trip eff ~94-97%.

Pure NumPy. No scipy, no ODEs, no AI.
"""
import numpy as np


class CapacitanceEsrCurve:
    def __init__(self, params):
        p = params
        self.C = float(p["capacitance"]["value"])
        self.esr = float(p["esr"]["value"])
        self.V_max = float(p["V_max"]["value"])
        self.V_min = float(p["V_min"]["value"])

    def voltage(self, charge_C):
        """Terminal open-circuit voltage for a stored charge Q (coulombs)."""
        return np.asarray(charge_C, dtype=float) / self.C

    def energy(self, voltage):
        """Stored energy 0.5*C*V^2 (joules)."""
        v = np.asarray(voltage, dtype=float)
        return 0.5 * self.C * v * v

    def usable_energy(self):
        return 0.5 * self.C * (self.V_max ** 2 - self.V_min ** 2)

    def roundtrip_efficiency(self, current):
        """DC round-trip efficiency at constant |I|: (V-IR)/(V+IR) at V_max."""
        i = np.abs(np.asarray(current, dtype=float))
        drop = i * self.esr
        return np.clip((self.V_max - drop) / (self.V_max + drop), 0.0, 1.0)
