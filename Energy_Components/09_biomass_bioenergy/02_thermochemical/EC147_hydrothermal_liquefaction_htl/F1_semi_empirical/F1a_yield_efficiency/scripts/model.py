"""
EC147 -- Hydrothermal Liquefaction (HTL) -- F1a Yield Model

Simplest semi-empirical model at fixed operating point (350 degC, 200 bar):
    bio_crude_kg = feedstock_dry_kg * bio_crude_yield

Key advantage: moisture-tolerant — no need to pre-dry wet biomass/algae.
F1b adds feedstock-type corrections (protein/lipid/carbohydrate), temperature effects.

References:
    Toor, S.S. et al. (2011). Energy 36:2328.
    Elliott, D.C. et al. (2015). Bioresour. Technol. 178:147.
    Biller, P. & Ross, A.B. (2011). Bioresour. Technol. 102:215.
"""

import numpy as np


class HTLF1a:
    """HTL reactor: bio_crude = feedstock_dry * bio_crude_yield."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.bio_crude_yield  = float(u["bio_crude_yield"]["value"])
        self.LHV_bio_crude    = float(u["LHV_bio_crude_MJ_kg"]["value"])
        self.T_operating      = float(u["T_operating_degC"]["value"])
        self.P_operating      = float(u["P_operating_bar"]["value"])
        self.aqueous_yield    = float(u["aqueous_yield"]["value"])
        self.gas_yield        = float(u["gas_yield"]["value"])
        self.solid_yield      = float(u["solid_yield"]["value"])
        self.SEC_kWh_per_t    = float(u["SEC_kWh_per_t"]["value"])

    def predict(self, feedstock_dry_kg_per_h: float) -> dict:
        """
        Parameters
        ----------
        feedstock_dry_kg_per_h : float  Dry feedstock input [kg/h]

        Returns
        -------
        dict with:
            bio_crude_kg_per_h    : bio-crude output [kg/h]
            aqueous_kg_per_h      : aqueous phase [kg/h]
            gas_kg_per_h          : gas phase (CO2-rich) [kg/h]
            solid_kg_per_h        : solid residue [kg/h]
            energy_output_MW      : energy in bio-crude [MW]
            electricity_kW        : parasitic electrical load [kW]
        """
        feed = float(np.clip(feedstock_dry_kg_per_h, 0.0, None))

        bio_crude = feed * self.bio_crude_yield
        aqueous   = feed * self.aqueous_yield
        gas       = feed * self.gas_yield
        solid     = feed * self.solid_yield

        energy_out = bio_crude * self.LHV_bio_crude / 3600.0   # MW
        electricity = feed / 1000.0 * self.SEC_kWh_per_t       # kW (feed in t/h)

        return {
            "bio_crude_kg_per_h": float(bio_crude),
            "aqueous_kg_per_h":   float(aqueous),
            "gas_kg_per_h":       float(gas),
            "solid_kg_per_h":     float(solid),
            "energy_output_MW":   float(energy_out),
            "electricity_kW":     float(electricity),
        }
