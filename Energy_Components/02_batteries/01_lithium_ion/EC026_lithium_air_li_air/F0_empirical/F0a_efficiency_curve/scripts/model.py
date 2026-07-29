"""
EC026 -- Lithium-Air (Li-Air) Battery -- F0 Empirical Round-Trip Efficiency Curve

F0a: the simplest faithful battery model -- a tabulated round-trip efficiency
curve vs C-rate, served by 1-D np.interp over breakpoints. The curve is derived
from the cell's own ohmic loss fraction (I*R / V_nom) using the SAME R_internal,
nominal voltage and capacity reused from the F1a semi-empirical model.

    eta_rt(C) = ((1 - f) / (1 + f)),   f = C * capacity * R_internal / V_nom

Empirical lookup only -- NumPy, no ODEs, no AI.

Reference (reused from F1a):
    Li-Air F1a defaults (Q_nom=10 Ah, R_int=200 mOhm, flat OCV ~2.96 V from O2 reduction)
"""

import numpy as np


class LiAirBatteryF0a:
    """Round-trip-efficiency lookup model for the EC026 cell."""

    def __init__(self, params: dict):
        cell = params["cell"]
        curve = params["efficiency_curve"]
        self.nominal_voltage = cell["nominal_voltage"]["value"]
        self.capacity = cell["capacity"]["value"]
        self.r_internal = cell["internal_resistance"]["value"]
        self.crate_bp = np.asarray(curve["c_rate"]["value"], dtype=float)
        self.eta_bp = np.asarray(curve["round_trip_efficiency"]["value"], dtype=float)
        self.crate_max = float(self.crate_bp[-1])

    def round_trip_efficiency(self, c_rate):
        """Round-trip efficiency (0-1) at a given C-rate, via 1-D lookup."""
        c = np.clip(np.asarray(c_rate, dtype=float), 0.0, self.crate_max)
        return np.interp(c, self.crate_bp, self.eta_bp)

    def usable_energy(self, c_rate, energy_in_wh):
        """Energy delivered after a round trip of `energy_in_wh` Wh at `c_rate`."""
        return np.asarray(energy_in_wh, dtype=float) * self.round_trip_efficiency(c_rate)


if __name__ == "__main__":
    import json, os
    p = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
    m = LiAirBatteryF0a(json.load(open(p)))
    for c in (0.0, 0.5, m.crate_max):
        print(f"C={c:.2f}  eta_rt={m.round_trip_efficiency(c):.4f}")
