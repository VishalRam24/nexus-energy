"""
EC205 — CO2 Electrolyzer — F1a Energy Model

CO2 reduction reaction (CO2RR) to CO via electrochemical cell.
SEC = 8 kWh/kgCO2 at design conditions (V_cell=3V, FE=0.85).

Model:
    CO_produced [kg/h] = capacity_fraction * capacity_kgCO2_h * faradaic_efficiency * (M_CO/M_CO2)
    W_elec [kWh/h]     = CO2_processed * SEC
    CO2_converted [kg/h] = capacity_fraction * capacity_kgCO2_h * faradaic_efficiency

    SEC varies with current density (simplified linear):
    SEC(j) = SEC_ref * (1 + 0.001 * |j - j_ref|)

References:
    Jouny, M. et al. (2018). General Techno-Economic Analysis of CO2 Electrolysis Systems.
      Ind. Eng. Chem. Res. 57(6):2165-2177.
    Bushuyev, O.S. et al. (2018). What Should We Make with CO2? Joule 2(5):825-832.
"""

import numpy as np

M_CO2 = 44.01   # g/mol
M_CO = 28.01    # g/mol
RATIO_CO_CO2 = M_CO / M_CO2


class CO2ElectrolyzerF1a:
    """CO2 electrolyzer energy and conversion model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.FE = u["faradaic_efficiency"]["value"]
        self.V_cell = u["V_cell"]["value"]
        self.SEC_ref = u["SEC_kWh_kgCO2"]["value"]    # kWh/kgCO2
        self.j_ref = u["current_density_mA_cm2"]["value"]
        self.capacity_kgCO2_h = 1000.0  # 1 tCO2/h design

    def sec_kWh_kgCO2(self, current_density=None):
        """Specific energy consumption [kWh/kgCO2]."""
        j = self.j_ref if current_density is None else np.asarray(current_density, dtype=float)
        return self.SEC_ref * (1.0 + 0.001 * np.abs(j - self.j_ref))

    def co2_converted(self, capacity_fraction):
        """CO2 electrochemically converted [kg/h]."""
        cf = np.asarray(capacity_fraction, dtype=float)
        return cf * self.capacity_kgCO2_h * self.FE

    def co_produced(self, capacity_fraction):
        """CO produced [kg/h]."""
        return self.co2_converted(capacity_fraction) * RATIO_CO_CO2

    def electric_energy(self, capacity_fraction, current_density=None):
        """Electrical energy consumption [kWh/h]."""
        co2 = cf_kg = np.asarray(capacity_fraction, dtype=float) * self.capacity_kgCO2_h
        sec = self.sec_kWh_kgCO2(current_density)
        return co2 * sec
