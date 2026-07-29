"""
EC028 -- Lead-Acid Battery -- F0 Empirical Round-Trip Efficiency Curve

F0a: the simplest faithful battery model -- a tabulated round-trip efficiency
curve vs C-rate, served by 1-D np.interp over breakpoints. The curve is derived
from the cell's own ohmic loss fraction (I*R / V_nom) using the SAME R_internal,
nominal voltage and capacity reused from the F1a semi-empirical model.

    eta_rt(C) = ((1 - f) / (1 + f)),   f = C * capacity * R_internal / V_nom

Empirical lookup only -- NumPy, no ODEs, no AI.

Reference (reused from F1a):
    Copetti et al. (1993), J. Power Sources 47(1):109-118; Manwell & McGowan (1993), Sol. Energy 50(5):399-405
"""

import numpy as np


class LeadAcidBatteryF0a:
    """Round-trip-efficiency lookup model for the EC028 cell."""

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
    m = LeadAcidBatteryF0a(json.load(open(p)))
    for c in (0.0, 0.5, m.crate_max):
        print(f"C={c:.2f}  eta_rt={m.round_trip_efficiency(c):.4f}")
