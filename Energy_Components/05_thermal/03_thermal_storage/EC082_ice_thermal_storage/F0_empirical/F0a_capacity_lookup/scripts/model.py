"""F0a empirical capacity lookup for Ice Thermal Storage.

Data source: ASHRAE (2020) Handbook ch.51; Dincer & Rosen (2021)
Usable stored energy is a linear lookup of state-of-charge:
    E_usable = SOC * E_capacity
Round-trip efficiency and self-discharge are constant empirical ratings. NumPy only.
"""
import numpy as np


class CapacityLookup:
    def __init__(self, soc_bp, energy_bp):
        self.soc = np.asarray(soc_bp, dtype=float)
        self.energy = np.asarray(energy_bp, dtype=float)

    def energy_at(self, soc):
        s = float(np.clip(soc, self.soc[0], self.soc[-1]))
        return float(np.interp(s, self.soc, self.energy))
