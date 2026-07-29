"""
EC206 — CO2 Mineralization — F1a Energy Model

Permanent CO2 storage: MgO/CaO + CO2 -> MgCO3/CaCO3 (carbonates).
Exothermic reaction; main energy input is mechanical (grinding, mixing, pumping).

Model:
    CO2_stored [tCO2/h]    = capacity_fraction * capacity_tCO2_h * conversion
    W_elec [GJ/h]          = CO2_stored * SEC
    carbonate_produced [t/h] = CO2_stored * carbonate_yield

References:
    Sanna, A. et al. (2014). A review of mineral carbonation technologies to sequester CO2.
      Prog. Energy Combust. Sci. 44:35-61.
    IPCC (2005). Carbon Dioxide Capture and Storage. Cambridge University Press.
"""

import numpy as np


class CO2MineralizationF1a:
    """CO2 mineralization energy and conversion model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.conversion = u["conversion"]["value"]
        self.SEC = u["SEC_GJ_tCO2"]["value"]              # GJ/tCO2
        self.carbonate_yield = u["carbonate_yield_kg_kg"]["value"]  # kg carbonate/kg CO2
        self.capacity_tCO2_h = 1.0

    def co2_stored(self, capacity_fraction):
        """CO2 permanently stored [tCO2/h]."""
        cf = np.asarray(capacity_fraction, dtype=float)
        return cf * self.capacity_tCO2_h * self.conversion

    def electric_energy(self, capacity_fraction):
        """Electrical energy consumption [GJ/h]."""
        return self.co2_stored(capacity_fraction) * self.SEC

    def carbonate_produced(self, capacity_fraction):
        """Carbonate product mass [t/h]."""
        # carbonate_yield in kg/kg = t/t
        return self.co2_stored(capacity_fraction) * self.carbonate_yield
