"""F0a empirical round-trip-efficiency lookup for Iron-Air Battery (EC033).

F0 = simplest fidelity: a tabulated DC round-trip efficiency vs C-rate curve.
The curve is the voltaic efficiency eta = (V-IR)/(V+IR) from the cell's ohmic
overpotential, normalised to the literature rated round-trip efficiency.

Data source (reused from EC033 F1a):
EC033 F1a (V_nom=1.28 V, R_int=100 mOhm, Q_nom=4.0 Ah); Fe+O2->Fe2O3. Round-trip eff ~40-55% (Weber et al., J. Appl. Electrochem. 2011).

Pure NumPy. No scipy, no ODEs, no AI.
"""
import numpy as np


class RoundtripEfficiencyCurve:
    def __init__(self, params):
        p = params
        self.c_rates = np.asarray(p["c_rate_breakpoints"]["value"], dtype=float)
        self.eta = np.asarray(p["roundtrip_efficiency"]["value"], dtype=float)
        self.V_nom = float(p["V_nominal"]["value"])
        self.R = float(p["R_series"]["value"])
        self.Q_nom = float(p["Q_nominal"]["value"])
        self.eta_rated = float(p["eta_rated"]["value"])
        self.c_rated = float(p["c_rated"]["value"])

    def efficiency(self, c_rate):
        """DC round-trip efficiency at a given (abs) C-rate, via 1-D interpolation."""
        c = np.abs(np.asarray(c_rate, dtype=float))
        return np.interp(c, self.c_rates, self.eta)

    def loss_fraction(self, c_rate):
        return 1.0 - self.efficiency(c_rate)
