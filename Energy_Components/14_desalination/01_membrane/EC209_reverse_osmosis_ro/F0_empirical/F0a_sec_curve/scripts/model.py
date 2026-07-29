"""EC209 RO F0a — empirical SEC-vs-recovery lookup curve.

Specific Energy Consumption (kWh/m3) of seawater RO rises with recovery as the
reject brine concentrates and required feed pressure climbs. A 1-D np.interp
over a tabulated (recovery, SEC) breakpoint array.

Source: Elimelech & Phillip (2011), Science 333:712-717 (reused from EC209 F1a).
NumPy only.
"""
import numpy as np


class SECCurve:
    def __init__(self, recovery_bp, sec_bp, recovery_rated, sec_rated, rejection):
        self.recovery_bp = np.asarray(recovery_bp, dtype=float)
        self.sec_bp = np.asarray(sec_bp, dtype=float)
        self.recovery_rated = float(recovery_rated)
        self.sec_rated = float(sec_rated)
        self.rejection = float(rejection)

    def sec(self, recovery):
        """SEC (kWh/m3) at given recovery; clamps to breakpoint endpoints."""
        return np.interp(recovery, self.recovery_bp, self.sec_bp)

    def permeate_salinity(self, recovery, feed_salinity):
        """Permeate salinity (g/L) from salt rejection."""
        return np.asarray(feed_salinity, dtype=float) * (1.0 - self.rejection)
