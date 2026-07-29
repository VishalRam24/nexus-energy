"""
EC210 — Electrodialysis (ED) — F1a SEC, Recovery, Rejection

Ion-exchange membrane process driven by DC voltage. Ions migrate through alternating
cation/anion membranes under electric field.

Model:
    SEC(salinity) = SEC_ref * (salinity / salinity_ref)^0.7  — sublinear scaling
    permeate_flow = feed_flow * recovery
    W_elec = permeate_flow * SEC

References:
    Strathmann, H. (2004). Ion-Exchange Membrane Separation Processes. Elsevier.
    Burn, S. et al. (2015). Desalination techniques — A review of the opportunities
      for desalination in agriculture. Desalination 364:2-16.
"""

import numpy as np

SALINITY_REF = 4000.0  # ppm TDS, design point


class ElectrodialysisF1a:
    """ED SEC, recovery, rejection model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.SEC_ref = u["SEC_kWh_m3"]["value"]       # kWh/m3
        self.recovery = u["recovery"]["value"]
        self.rejection = u["rejection"]["value"]
        self.V_cell = u["V_cell_pair_V"]["value"]
        self.n_cells = u["n_cell_pairs"]["value"]
        self.capacity_m3_h = u["capacity_m3_h"]["value"]

    def sec_kWh_m3(self, feed_salinity_ppm=4000.0):
        """SEC [kWh/m3 permeate] as function of feed salinity."""
        sal = np.asarray(feed_salinity_ppm, dtype=float)
        return self.SEC_ref * (sal / SALINITY_REF) ** 0.7

    def permeate_flow(self, capacity_fraction):
        """Permeate flow rate [m3/h]."""
        cf = np.asarray(capacity_fraction, dtype=float)
        return cf * self.capacity_m3_h * self.recovery

    def electric_power(self, capacity_fraction, feed_salinity_ppm=4000.0):
        """Electrical power [kWh/h = kW]."""
        perm = self.permeate_flow(capacity_fraction)
        sec = self.sec_kWh_m3(feed_salinity_ppm)
        return perm * sec

    def concentrate_flow(self, capacity_fraction):
        """Concentrate (reject) flow [m3/h]."""
        cf = np.asarray(capacity_fraction, dtype=float)
        feed = cf * self.capacity_m3_h
        perm = self.permeate_flow(capacity_fraction)
        return feed - perm
