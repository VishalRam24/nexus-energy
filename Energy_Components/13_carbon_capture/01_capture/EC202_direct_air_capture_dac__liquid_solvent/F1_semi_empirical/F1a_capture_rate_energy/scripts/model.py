"""
EC202 — Direct Air Capture (DAC) Liquid Solvent — F1a Capture Rate & Energy

KOH liquid solvent DAC. Two-step process: absorption (air contactor) + regeneration (calciner).
Capture rate fixed at 0.90; SEC scales linearly with capacity fraction.

Model:
    CO2_captured [tCO2/h] = capacity_fraction * capacity_tCO2_h * capture_rate
    Q_thermal [GJ/h]      = CO2_captured * SEC_thermal
    W_elec [GJ/h]         = CO2_captured * SEC_elec
    SEC_total [GJ/tCO2]   = SEC_thermal + SEC_elec

References:
    Keith, D.W. et al. (2018). A Process for Capturing CO2 from the Atmosphere.
      Joule 2(8):1573-1594.
    Fasihi, M. et al. (2019). Techno-Economic Assessment of CO2 Direct Air Capture Plants.
      J. Cleaner Production 224:957-980.
"""

import numpy as np


class DACF1a:
    """DAC liquid solvent capture rate and energy model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.capture_rate = u["capture_rate"]["value"]
        self.SEC_thermal = u["SEC_thermal_GJ_tCO2"]["value"]  # GJ/tCO2
        self.SEC_elec = u["SEC_elec_GJ_tCO2"]["value"]        # GJ/tCO2
        self.T_regen = u["T_regen_C"]["value"]
        # Default design capacity: 1 tCO2/h (user scales)
        self.capacity_tCO2_h = 1.0

    def co2_captured(self, capacity_fraction):
        """CO2 captured [tCO2/h] at given capacity fraction."""
        cf = np.asarray(capacity_fraction, dtype=float)
        return cf * self.capacity_tCO2_h * self.capture_rate

    def thermal_energy(self, capacity_fraction):
        """Thermal energy consumption [GJ/h]."""
        return self.co2_captured(capacity_fraction) * self.SEC_thermal

    def electric_energy(self, capacity_fraction):
        """Electrical energy consumption [GJ/h]."""
        return self.co2_captured(capacity_fraction) * self.SEC_elec

    def sec_total(self):
        """Total specific energy consumption [GJ/tCO2]."""
        return self.SEC_thermal + self.SEC_elec
