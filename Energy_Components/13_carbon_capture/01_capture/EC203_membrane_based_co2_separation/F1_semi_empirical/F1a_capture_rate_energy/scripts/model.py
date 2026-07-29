"""
EC203 — Membrane-Based CO2 Separation — F1a Capture Rate & Energy

Pressure-driven membrane separation of CO2 from flue gas or atmosphere.
SEC depends on pressure ratio (feed/permeate). Recovery and purity fixed.

Model:
    SEC(PR) = SEC_ref + SEC_pressure_coeff * max(PR - PR_ref, 0)
    SEC clamped to [SEC_min, SEC_max]

    CO2_captured [tCO2/h] = capacity_fraction * capacity_tCO2_h * CO2_recovery
    W_elec [GJ/h]         = CO2_captured * SEC [GJ/tCO2]

References:
    Baker, R.W. (2002). Future Directions of Membrane Gas Separation Technology.
      Ind. Eng. Chem. Res. 41(6):1393-1411.
    Merkel, T.C. et al. (2010). Power plant post-combustion carbon dioxide capture:
      An opportunity for membranes. J. Membrane Science 359:126-139.
"""

import numpy as np


class MembraneF1a:
    """Membrane CO2 separation: capture rate and SEC model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.CO2_recovery = u["CO2_recovery"]["value"]
        self.CO2_purity = u["CO2_purity"]["value"]
        self.SEC_ref = u["SEC_ref_MJ_kgCO2"]["value"]    # MJ/kgCO2
        self.PR_ref = u["pressure_ratio"]["value"]
        self.SEC_min = u["SEC_min_MJ_kgCO2"]["value"]
        self.SEC_max = u["SEC_max_MJ_kgCO2"]["value"]
        self.SEC_coeff = u["SEC_pressure_coeff"]["value"]
        self.capacity_tCO2_h = 1.0  # design capacity, user-scaled

    def sec_MJ_kgCO2(self, pressure_ratio=None):
        """Specific energy consumption [MJ/kgCO2]."""
        PR = self.PR_ref if pressure_ratio is None else np.asarray(pressure_ratio, dtype=float)
        sec = self.SEC_ref + self.SEC_coeff * np.maximum(PR - self.PR_ref, 0.0)
        return np.clip(sec, self.SEC_min, self.SEC_max)

    def co2_captured(self, capacity_fraction):
        """CO2 captured [tCO2/h]."""
        cf = np.asarray(capacity_fraction, dtype=float)
        return cf * self.capacity_tCO2_h * self.CO2_recovery

    def electric_energy(self, capacity_fraction, pressure_ratio=None):
        """Electrical energy [GJ/h]."""
        co2 = self.co2_captured(capacity_fraction)
        sec = self.sec_MJ_kgCO2(pressure_ratio)  # MJ/kgCO2 = GJ/tCO2
        return co2 * sec  # GJ/h
