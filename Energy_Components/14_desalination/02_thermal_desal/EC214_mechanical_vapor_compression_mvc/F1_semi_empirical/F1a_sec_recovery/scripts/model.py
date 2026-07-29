"""
EC214 — Mechanical Vapor Compression (MVC) — F1a SEC, Recovery

All-electric desalination: compressor raises vapor temperature/pressure to drive heat exchange.
No external thermal input required (self-contained heat pump cycle).

Model:
    SEC(CR) = SEC_ref * CR / CR_ref / eta_comp  — scales with compression ratio
    SEC clamped to [SEC_min, SEC_max]
    distillate_flow = capacity_fraction * capacity_m3_h
    W_elec = distillate_flow * SEC

References:
    Mistry, K.H. et al. (2011). Effect of Entropy Generation on the Energy Efficiency
      of Single-Effect Absorption and Single-Effect MVC. Entropy 13(10):1829-1864.
"""

import numpy as np


class MVCF1a:
    """MVC desalination SEC and recovery model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.SEC_ref = u["SEC_ref_kWh_m3"]["value"]
        self.SEC_min = u["SEC_min_kWh_m3"]["value"]
        self.SEC_max = u["SEC_max_kWh_m3"]["value"]
        self.recovery = u["recovery"]["value"]
        self.GOR_equiv = u["GOR_equiv"]["value"]
        self.capacity_m3_h = u["capacity_m3_h"]["value"]
        self.CR_ref = u["compression_ratio"]["value"]
        self.eta_comp = u["compressor_efficiency"]["value"]

    def sec_kWh_m3(self, compression_ratio=None):
        """SEC [kWh/m3] as function of compression ratio."""
        CR = self.CR_ref if compression_ratio is None else np.asarray(compression_ratio, dtype=float)
        sec = self.SEC_ref * (CR / self.CR_ref) / self.eta_comp * self.eta_comp
        # Simplified: SEC proportional to CR
        sec = self.SEC_ref * (CR / self.CR_ref)
        return np.clip(sec, self.SEC_min, self.SEC_max)

    def distillate_flow(self, capacity_fraction):
        """Distillate flow [m3/h]."""
        cf = np.asarray(capacity_fraction, dtype=float)
        return cf * self.capacity_m3_h

    def electric_power(self, capacity_fraction, compression_ratio=None):
        """Electrical power [kWh/h]."""
        dist = self.distillate_flow(capacity_fraction)
        sec = self.sec_kWh_m3(compression_ratio)
        return dist * sec

    def concentrate_flow(self, capacity_fraction):
        """Concentrate flow [m3/h]."""
        cf = np.asarray(capacity_fraction, dtype=float)
        feed = cf * self.capacity_m3_h / self.recovery  # feed > distillate
        return feed - self.distillate_flow(capacity_fraction)
