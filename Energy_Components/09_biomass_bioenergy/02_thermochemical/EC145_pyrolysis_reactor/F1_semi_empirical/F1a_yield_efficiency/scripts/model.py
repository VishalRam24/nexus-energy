"""
EC145 -- Pyrolysis Reactor -- F1a Yield Model

Simplest semi-empirical model at fixed operating point (fast pyrolysis, 500 degC):
    bio_oil_kg  = feedstock_dry_kg * bio_oil_yield
    char_kg     = feedstock_dry_kg * char_yield
    gas_kg      = feedstock_dry_kg * gas_yield

Yields sum to 1.0 by design (mass balance at rated conditions).
F1b adds feedstock-specific yields, moisture-LHV coupling, and temperature effects.

References:
    Bridgwater, A.V. (2012). Biomass Bioenergy 38:68.
    Mohan, D. et al. (2006). Energy Fuels 20:848.
"""

import numpy as np


class PyrolysisReactorF1a:
    """Fast pyrolysis reactor: product = feedstock * yield_fraction."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.bio_oil_yield    = float(u["bio_oil_yield"]["value"])
        self.char_yield       = float(u["char_yield"]["value"])
        self.gas_yield        = float(u["gas_yield"]["value"])
        self.T_operating      = float(u["T_operating_degC"]["value"])
        self.LHV_bio_oil      = float(u["LHV_bio_oil_MJ_kg"]["value"])
        self.LHV_char         = float(u["LHV_char_MJ_kg"]["value"])
        self.LHV_gas          = float(u["LHV_gas_MJ_kg"]["value"])

    def predict(self, feedstock_dry_kg_per_h: float) -> dict:
        """
        Parameters
        ----------
        feedstock_dry_kg_per_h : float  Dry biomass feed rate [kg/h]

        Returns
        -------
        dict with:
            bio_oil_kg_per_h    : bio-oil production [kg/h]
            char_kg_per_h       : biochar production [kg/h]
            gas_kg_per_h        : syngas production [kg/h]
            energy_bio_oil_MW   : energy in bio-oil [MW]
            energy_char_MW      : energy in char [MW]
            energy_gas_MW       : energy in gas [MW]
            mass_balance_check  : bio_oil+char+gas fraction sum (should be ~1.0)
        """
        feed = float(np.clip(feedstock_dry_kg_per_h, 0.0, None))

        bio_oil = feed * self.bio_oil_yield
        char    = feed * self.char_yield
        gas     = feed * self.gas_yield

        # Energy [MJ/h -> MW: divide by 3600]
        e_oil  = bio_oil * self.LHV_bio_oil / 3600.0
        e_char = char    * self.LHV_char    / 3600.0
        e_gas  = gas     * self.LHV_gas     / 3600.0

        return {
            "bio_oil_kg_per_h":  float(bio_oil),
            "char_kg_per_h":     float(char),
            "gas_kg_per_h":      float(gas),
            "energy_bio_oil_MW": float(e_oil),
            "energy_char_MW":    float(e_char),
            "energy_gas_MW":     float(e_gas),
            "mass_balance_check": float(self.bio_oil_yield + self.char_yield + self.gas_yield),
        }
