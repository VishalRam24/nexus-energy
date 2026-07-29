"""EC211 FO F0a — empirical SEC lookup with optional draw regeneration.

Forward-osmosis SEC (kWh/m3) splits into a small membrane-pumping term (rises
mildly at part-load) plus an optional draw-solution regeneration term. 1-D
np.interp gives the part-load membrane multiplier.

Source: Lutchmiah et al. (2014); Zhao et al. (2012) (reused from EC211 F1a).
NumPy only.
"""
import numpy as np


class SECLookup:
    def __init__(self, load_bp, mem_factor_bp, sec_membrane, sec_regen,
                 sec_total, recovery, rejection, capacity_m3_h):
        self.load_bp = np.asarray(load_bp, dtype=float)
        self.mem_factor_bp = np.asarray(mem_factor_bp, dtype=float)
        self.sec_membrane = float(sec_membrane)
        self.sec_regen = float(sec_regen)
        self.sec_total = float(sec_total)
        self.recovery = float(recovery)
        self.rejection = float(rejection)
        self.capacity_m3_h = float(capacity_m3_h)

    def membrane_factor(self, load_fraction):
        return np.interp(load_fraction, self.load_bp, self.mem_factor_bp)

    def sec(self, load_fraction=1.0, include_regen=True):
        """SEC (kWh/m3); add regen term if include_regen."""
        sec = self.sec_membrane * self.membrane_factor(load_fraction)
        if include_regen:
            sec = sec + self.sec_regen
        return sec

    def permeate_flow(self, load_fraction):
        return self.capacity_m3_h * np.asarray(load_fraction, dtype=float) * self.recovery
