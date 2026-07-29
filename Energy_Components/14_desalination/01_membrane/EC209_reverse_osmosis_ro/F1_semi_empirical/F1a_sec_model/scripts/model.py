"""
EC209 — Reverse Osmosis (RO) — F1a Specific Energy Consumption Model

SEC = f(recovery, feed_salinity, feed_pressure)
  osmotic_pressure = 0.7 * S_feed            [bar, S in g/L — simplified van't Hoff]
  P_feed = osmotic_pressure / recovery + dP_membrane   [bar, required feed pressure]
  SEC_ideal = P_feed / (recovery * 36)       [kWh/m3, conversion: 1 bar*m3 = 1/36 kWh]
  SEC_actual = SEC_ideal / (eta_pump * eta_ERD)
  With ERD: brine energy recovered = P_brine * (1-recovery) * eta_ERD
             P_brine ≈ P_feed - dP_membrane
  Net SEC = (P_feed - P_brine*(1-recovery)*eta_ERD) / (eta_pump * recovery * 36)

Reference:
    Elimelech, M. & Phillip, W. A. (2011). Science, 333, 712-717.
"""

import numpy as np


class ROF1a:
    """Reverse osmosis SEC model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta_pump = u["eta_pump"]["value"]
        self.eta_ERD = u["eta_ERD"]["value"]
        self.dP_membrane = u["dP_membrane"]["value"]      # bar
        self.pi_coeff = u["osmotic_pressure_coeff"]["value"]   # bar per g/L
        self.rejection = u["permeate_salinity_rejection"]["value"]

    def osmotic_pressure(self, S_feed):
        """Osmotic pressure [bar] from feed salinity [g/L]."""
        return self.pi_coeff * np.asarray(S_feed, dtype=float)

    def feed_pressure(self, S_feed, recovery):
        """Required feed pressure [bar]."""
        pi = self.osmotic_pressure(S_feed)
        r = np.asarray(recovery, dtype=float)
        return pi / r + self.dP_membrane

    def sec(self, S_feed, recovery):
        """Specific energy consumption [kWh/m3 permeate].

        Uses energy recovery device (ERD) on the brine stream.
        Conversion: 1 bar * m3 = 0.02778 kWh  (= 1/36)
        """
        r = np.asarray(recovery, dtype=float)
        P_feed = self.feed_pressure(S_feed, r)
        # Brine back-pressure (approx P_feed minus membrane loss)
        P_brine = P_feed - self.dP_membrane
        # Net hydraulic work [bar per m3 permeate]:
        #   High-pressure work on feed (per m3 feed) = P_feed / eta_pump
        #   Recovered from brine per m3 feed = P_brine * (1-r) * eta_ERD
        #   Per m3 permeate divide by r
        numerator = P_feed / self.eta_pump - P_brine * (1.0 - r) * self.eta_ERD
        sec_bar = numerator / r         # bar per m3 permeate
        sec_kwh = sec_bar / 36.0        # kWh/m3 permeate
        return np.clip(sec_kwh, 0.5, 20.0)

    def permeate_salinity(self, S_feed):
        """Permeate salinity [g/L] based on salt rejection."""
        return np.asarray(S_feed, dtype=float) * (1.0 - self.rejection)

    def compute(self, feed_salinity, recovery, feed_flow_m3h):
        """Full computation for given operating conditions.

        Parameters
        ----------
        feed_salinity : float or array  [g/L]
        recovery      : float or array  [0–1]
        feed_flow_m3h : float or array  [m3/hr]

        Returns
        -------
        dict: sec_kwhm3, permeate_flow_m3h, feed_pressure_bar, permeate_salinity_gl
        """
        S = np.asarray(feed_salinity, dtype=float)
        r = np.asarray(recovery, dtype=float)
        Q_feed = np.asarray(feed_flow_m3h, dtype=float)

        sec = self.sec(S, r)
        P_feed = self.feed_pressure(S, r)
        Q_perm = Q_feed * r
        S_perm = self.permeate_salinity(S)

        return {
            "sec_kwhm3": sec,
            "permeate_flow_m3h": Q_perm,
            "feed_pressure_bar": P_feed,
            "permeate_salinity_gl": S_perm,
        }
