"""EC214 MVC F0a — empirical SEC-vs-compression-ratio curve.

All-electric mechanical vapor compression: specific energy consumption
(kWh/m3) rises with vapor compression ratio because a larger ratio means a
larger temperature lift and more compressor work. 1-D np.interp over a
tabulated (CR, SEC) breakpoint array.

Source: Mistry et al. (2011) Entropy; GWI DesalData (reused from EC214 F1a).
NumPy only.
"""
import numpy as np


class SECCurve:
    def __init__(self, cr_bp, sec_bp, cr_rated, sec_rated,
                 sec_min, sec_max, recovery, capacity_m3_h):
        self.cr_bp = np.asarray(cr_bp, dtype=float)
        self.sec_bp = np.asarray(sec_bp, dtype=float)
        self.cr_rated = float(cr_rated)
        self.sec_rated = float(sec_rated)
        self.sec_min = float(sec_min)
        self.sec_max = float(sec_max)
        self.recovery = float(recovery)
        self.capacity_m3_h = float(capacity_m3_h)

    def sec(self, compression_ratio):
        """SEC (kWh/m3) at given compression ratio; clamps to endpoints."""
        return np.interp(compression_ratio, self.cr_bp, self.sec_bp)

    def distillate_flow(self, load_fraction):
        return self.capacity_m3_h * np.asarray(load_fraction, dtype=float)
