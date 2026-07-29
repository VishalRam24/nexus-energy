"""
EC143 — Biomass Gasifier — F1a Equilibrium Model

Syngas composition from equivalence ratio (ER) using simplified equilibrium approach.
Based on linear empirical fits calibrated against Zainal et al. (2001) equilibrium data.

Compositions at ER = 0.25 (design point):
    CO  = 0.22, H2 = 0.18, CO2 = 0.10, CH4 = 0.03, N2 = balance

Reference:
    Zainal, Z.A. et al. (2001). Prediction of performance of a downdraft gasifier
    using equilibrium modeling for different biomass materials.
    Energy Conversion and Management, 42(12), 1499-1515.
"""

import numpy as np


class BiomassGasifierF1a:
    """
    Biomass gasifier equilibrium model.
    Predicts syngas composition as a function of equivalence ratio and temperature.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.T_gasif = u["T_gasification"]["value"]        # degC
        self.LHV_biomass = u["LHV_biomass"]["value"]        # MJ/kg
        self.ER_design = u["ER_design"]["value"]
        self.LHV_CO = u["LHV_CO"]["value"]                  # MJ/Nm3
        self.LHV_H2 = u["LHV_H2"]["value"]                  # MJ/Nm3
        self.LHV_CH4 = u["LHV_CH4"]["value"]                # MJ/Nm3
        self.V_syngas = u["V_syngas_per_kg"]["value"]        # Nm3/kg_biomass

        # Reference fractions at ER_design
        self.CO_ref = u["CO_ref"]["value"]
        self.H2_ref = u["H2_ref"]["value"]
        self.CO2_ref = u["CO2_ref"]["value"]
        self.CH4_ref = u["CH4_ref"]["value"]

        # Sensitivity to ER
        self.dCO_dER = u["dCO_dER"]["value"]
        self.dH2_dER = u["dH2_dER"]["value"]
        self.dCO2_dER = u["dCO2_dER"]["value"]
        self.dCH4_dER = u["dCH4_dER"]["value"]

    def syngas_composition(self, ER, temperature_c=None):
        """
        Compute dry syngas mole fractions at given equivalence ratio.

        Parameters
        ----------
        ER : float or array
            Equivalence ratio (air-to-biomass / stoichiometric air-to-biomass), 0.2–0.5
        temperature_c : float or array, optional
            Gasification temperature (degC). Minor correction applied; default = design T.

        Returns
        -------
        dict with keys: CO, H2, CO2, CH4, N2 (all mole fractions, sum to 1.0)
        """
        ER = np.asarray(ER, dtype=float)
        dER = ER - self.ER_design

        CO  = np.clip(self.CO_ref  + self.dCO_dER  * dER, 0.0, 1.0)
        H2  = np.clip(self.H2_ref  + self.dH2_dER  * dER, 0.0, 1.0)
        CO2 = np.clip(self.CO2_ref + self.dCO2_dER * dER, 0.0, 1.0)
        CH4 = np.clip(self.CH4_ref + self.dCH4_dER * dER, 0.0, 1.0)

        # Temperature correction: higher T slightly shifts CO up, CH4 down
        if temperature_c is not None:
            T = np.asarray(temperature_c, dtype=float)
            dT = T - self.T_gasif  # degC above design T
            CO  = np.clip(CO  + 0.0002 * dT, 0.0, 1.0)
            CH4 = np.clip(CH4 - 0.0001 * dT, 0.0, 1.0)
            CO2 = np.clip(CO2 - 0.0001 * dT, 0.0, 1.0)
            H2  = np.clip(H2  + 0.0001 * dT, 0.0, 1.0)

        # N2 as balance to ensure fractions sum to 1
        combustibles = CO + H2 + CO2 + CH4
        N2 = np.clip(1.0 - combustibles, 0.0, 1.0)

        # Re-normalize to guarantee sum = 1
        total = CO + H2 + CO2 + CH4 + N2
        CO /= total; H2 /= total; CO2 /= total; CH4 /= total; N2 /= total

        return {"CO": CO, "H2": H2, "CO2": CO2, "CH4": CH4, "N2": N2}

    def lhv_syngas(self, ER, temperature_c=None):
        """
        Lower heating value of syngas in MJ/Nm3.
        LHV_syngas = LHV_CO*CO + LHV_H2*H2 + LHV_CH4*CH4
        """
        comp = self.syngas_composition(ER, temperature_c)
        return (self.LHV_CO * comp["CO"] +
                self.LHV_H2 * comp["H2"] +
                self.LHV_CH4 * comp["CH4"])

    def cold_gas_efficiency(self, ER, temperature_c=None, m_biomass=1.0):
        """
        Cold gas efficiency = LHV_syngas * V_syngas / (m_biomass * LHV_biomass).

        V_syngas is scaled by ER (more air → more N2 dilution, more volume but lower LHV).
        V_syngas_total ≈ V_syngas_ref * (ER / ER_design)^0.5 (weak scaling)
        """
        lhv = self.lhv_syngas(ER, temperature_c)
        ER = np.asarray(ER, dtype=float)
        # Syngas volume slightly increases with ER (more N2)
        V = self.V_syngas * (ER / self.ER_design) ** 0.5
        cge = (lhv * V) / (m_biomass * self.LHV_biomass)
        return np.clip(cge, 0.0, 1.0)
