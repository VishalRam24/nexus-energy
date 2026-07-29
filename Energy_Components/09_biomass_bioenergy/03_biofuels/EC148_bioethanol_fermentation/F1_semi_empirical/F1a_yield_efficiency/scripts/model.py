"""
EC148 -- Bioethanol Fermentation -- F1a Yield Model

Simplest semi-empirical model:
    ethanol_L  = sugar_input_kg * ethanol_yield_L_per_kg * eta_conversion
    ethanol_kg = ethanol_L * ethanol_density

Fermentation at 32 degC, 48h batch time.
F1b adds temperature-dependent kinetics, inhibition, multi-stage processing.

References:
    Balat, M. (2011). Energy Convers. Manage. 52:858.
    Mosier, N. et al. (2005). Bioresour. Technol. 96:673.
    Renewable Fuels Association (2020). Annual Industry Outlook.
"""

import numpy as np


class BioethanolFermentationF1a:
    """Bioethanol fermentation: ethanol = sugar * yield * eta_conversion."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.ethanol_yield  = float(u["ethanol_yield_L_per_kg_sugar"]["value"])  # L/kg_sugar
        self.eta_conversion = float(u["eta_conversion"]["value"])
        self.T_ferment      = float(u["T_fermentation_degC"]["value"])
        self.rho_ethanol    = float(u["ethanol_density_kg_per_L"]["value"])
        self.LHV_ethanol    = float(u["LHV_ethanol_MJ_kg"]["value"])
        self.CO2_ratio      = float(u["CO2_per_ethanol_kg_per_kg"]["value"])

    def predict(self, sugar_input_kg_per_h: float) -> dict:
        """
        Parameters
        ----------
        sugar_input_kg_per_h : float  Fermentable sugar feed [kg/h]

        Returns
        -------
        dict with:
            ethanol_L_per_h      : ethanol production [L/h]
            ethanol_kg_per_h     : ethanol mass flow [kg/h]
            CO2_kg_per_h         : CO2 co-product [kg/h]
            energy_output_MW     : energy content of ethanol [MW]
            eta_conversion       : conversion efficiency [-]
        """
        sugar = float(np.clip(sugar_input_kg_per_h, 0.0, None))

        ethanol_L  = sugar * self.ethanol_yield * self.eta_conversion
        ethanol_kg = ethanol_L * self.rho_ethanol
        CO2        = ethanol_kg * self.CO2_ratio
        energy_out = ethanol_kg * self.LHV_ethanol / 3600.0   # MW

        return {
            "ethanol_L_per_h":   float(ethanol_L),
            "ethanol_kg_per_h":  float(ethanol_kg),
            "CO2_kg_per_h":      float(CO2),
            "energy_output_MW":  float(energy_out),
            "eta_conversion":    float(self.eta_conversion),
        }
