"""
EC142 -- Biogas Upgrading to Biomethane -- F1a Yield Model

Simplest semi-empirical model:
    CH4_recovered_Nm3 = biogas_Nm3 * CH4_fraction_in * CH4_recovery
    electricity_kWh   = biogas_Nm3 * SEC_kWh_per_Nm3

Covers PSA, membrane, or amine scrubbing — all achieve similar overall
performance at the F1a level.

References:
    IEA Bioenergy Task 37 (2017). Biomethane - Status and Factors Affecting Market Development.
    Bauer, F. et al. (2013). Bioresour. Technol. 122:145.
"""

import numpy as np


class BiogasUpgradingF1a:
    """Biogas upgrading: biomethane = biogas * CH4_fraction * CH4_recovery."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.CH4_recovery        = float(u["CH4_recovery"]["value"])
        self.biomethane_purity   = float(u["biomethane_purity"]["value"])
        self.SEC_kWh_per_Nm3     = float(u["SEC_kWh_per_Nm3"]["value"])
        self.CO2_removal         = float(u["CO2_removal_fraction"]["value"])
        self.LHV_kWh_m3          = float(u["LHV_biomethane_kWh_m3"]["value"])

    def predict(self, biogas_flow_Nm3_per_h: float, CH4_fraction_in: float = 0.60) -> dict:
        """
        Parameters
        ----------
        biogas_flow_Nm3_per_h : float  Raw biogas flow rate [Nm3/h]
        CH4_fraction_in       : float  CH4 content of raw biogas [-]

        Returns
        -------
        dict with:
            biomethane_Nm3_per_h   : product biomethane flow [Nm3/h]
            CH4_recovered_Nm3_per_h: methane recovered [Nm3/h]
            electricity_kW         : electrical consumption [kW]
            energy_output_kW       : energy in product biomethane [kW]
            CH4_recovery           : methane recovery fraction [-]
            biomethane_purity      : product purity [-]
        """
        Q_raw = float(np.clip(biogas_flow_Nm3_per_h, 0.0, None))
        CH4_in = float(np.clip(CH4_fraction_in, 0.0, 1.0))

        CH4_recovered = Q_raw * CH4_in * self.CH4_recovery   # Nm3/h CH4
        # Biomethane volume at target purity
        biomethane    = CH4_recovered / self.biomethane_purity
        electricity   = Q_raw * self.SEC_kWh_per_Nm3          # kW (kWh/h)
        energy_out    = CH4_recovered * self.LHV_kWh_m3

        return {
            "biomethane_Nm3_per_h":    float(biomethane),
            "CH4_recovered_Nm3_per_h": float(CH4_recovered),
            "electricity_kW":          float(electricity),
            "energy_output_kW":        float(energy_out),
            "CH4_recovery":            float(self.CH4_recovery),
            "biomethane_purity":       float(self.biomethane_purity),
        }
