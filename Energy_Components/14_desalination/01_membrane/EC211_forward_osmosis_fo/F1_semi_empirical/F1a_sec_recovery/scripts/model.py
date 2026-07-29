"""
EC211 — Forward Osmosis (FO) — F1a SEC, Recovery, Rejection

Osmotically driven membrane process: water migrates from low-concentration feed
to high-concentration draw solution through semi-permeable membrane.
Draw solution regeneration is a major energy consumer.

Model:
    SEC_total = SEC_membrane + SEC_regen  (when regen included)
    permeate_flow = feed_flow * recovery
    W_elec = permeate_flow * SEC_total

References:
    Lutchmiah, K. et al. (2014). Forward osmosis for application in wastewater treatment:
      A review. Water Res. 58:179-197.
    Zhao, S. et al. (2012). Recent developments in forward osmosis: Opportunities and
      challenges. J. Membrane Sci. 396:1-21.
"""

import numpy as np


class ForwardOsmosisF1a:
    """FO SEC, recovery, rejection model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.SEC_membrane = u["SEC_membrane_kWh_m3"]["value"]
        self.SEC_regen = u["SEC_regen_kWh_m3"]["value"]
        self.recovery = u["recovery"]["value"]
        self.rejection = u["rejection"]["value"]
        self.capacity_m3_h = u["capacity_m3_h"]["value"]

    def sec_kWh_m3(self, include_regen=True):
        """Total SEC [kWh/m3 permeate]."""
        if include_regen:
            return self.SEC_membrane + self.SEC_regen
        return self.SEC_membrane

    def permeate_flow(self, capacity_fraction):
        """Permeate (product water) flow [m3/h]."""
        cf = np.asarray(capacity_fraction, dtype=float)
        return cf * self.capacity_m3_h * self.recovery

    def electric_power(self, capacity_fraction, include_regen=True):
        """Electrical power [kWh/h]."""
        perm = self.permeate_flow(capacity_fraction)
        sec = self.sec_kWh_m3(include_regen)
        return perm * sec

    def concentrate_flow(self, capacity_fraction):
        """Concentrate flow [m3/h]."""
        cf = np.asarray(capacity_fraction, dtype=float)
        feed = cf * self.capacity_m3_h
        return feed - self.permeate_flow(capacity_fraction)
