"""EC210 ED F0a — empirical SEC-vs-load lookup.

Brackish-water electrodialysis SEC (kWh/m3) is ~constant near rated load and
rises mildly at part-load due to fixed stack/pumping losses. 1-D np.interp over
a tabulated (load_fraction, SEC) breakpoint array.

Source: Strathmann (2004); GWI DesalData (reused from EC210 F1a). NumPy only.
"""
import numpy as np


class SECLookup:
    def __init__(self, load_bp, sec_bp, sec_rated, recovery, rejection, capacity_m3_h):
        self.load_bp = np.asarray(load_bp, dtype=float)
        self.sec_bp = np.asarray(sec_bp, dtype=float)
        self.sec_rated = float(sec_rated)
        self.recovery = float(recovery)
        self.rejection = float(rejection)
        self.capacity_m3_h = float(capacity_m3_h)

    def sec(self, load_fraction):
        """SEC (kWh/m3) at given load fraction; clamps to endpoints."""
        return np.interp(load_fraction, self.load_bp, self.sec_bp)

    def permeate_flow(self, load_fraction):
        """Permeate (product) flow m3/h = capacity * load * recovery."""
        return self.capacity_m3_h * np.asarray(load_fraction, dtype=float) * self.recovery
