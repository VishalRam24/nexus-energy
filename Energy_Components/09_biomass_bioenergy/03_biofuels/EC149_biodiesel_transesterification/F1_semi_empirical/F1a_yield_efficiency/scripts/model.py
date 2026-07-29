"""
EC149 -- Biodiesel Transesterification -- F1a Yield Model

Simplest semi-empirical model (alkali-catalyzed, 60 degC):
    FAME_kg    = oil_input_kg * FAME_yield
    glycerol_kg = oil_input_kg * glycerol_yield
    methanol_consumed_kg = oil_input_kg * methanol_consumed_ratio

Methanol:oil molar ratio = 6:1 (3x stoichiometric excess for >95% conversion).
F1b adds temperature/catalyst effects, FFA content correction.

References:
    Freedman, B. et al. (1984). JAOCS 61:1638.
    Ma, F. & Hanna, M.A. (1999). Bioresour. Technol. 70:1.
    van Gerpen, J. (2005). Fuel Process. Technol. 86:1097.
"""

import numpy as np


class BiodieselTransesterificationF1a:
    """Transesterification: FAME = oil * FAME_yield."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.FAME_yield         = float(u["FAME_yield_kg_per_kg_oil"]["value"])
        self.glycerol_yield     = float(u["glycerol_yield_kg_per_kg_oil"]["value"])
        self.MeOH_consumed      = float(u["methanol_consumed_kg_per_kg_oil"]["value"])
        self.T_reaction         = float(u["T_reaction_degC"]["value"])
        self.LHV_FAME           = float(u["LHV_FAME_MJ_kg"]["value"])
        self.density_FAME       = float(u["density_FAME_kg_per_L"]["value"])

    def predict(self, oil_input_kg_per_h: float) -> dict:
        """
        Parameters
        ----------
        oil_input_kg_per_h : float  Vegetable oil / waste oil feed [kg/h]

        Returns
        -------
        dict with:
            FAME_kg_per_h          : biodiesel (FAME) production [kg/h]
            FAME_L_per_h           : biodiesel volume [L/h]
            glycerol_kg_per_h      : glycerol co-product [kg/h]
            methanol_consumed_kg_per_h : methanol consumption [kg/h]
            energy_output_MW       : energy content of FAME [MW]
            FAME_yield             : mass yield [-]
        """
        oil = float(np.clip(oil_input_kg_per_h, 0.0, None))

        FAME        = oil * self.FAME_yield
        glycerol    = oil * self.glycerol_yield
        MeOH        = oil * self.MeOH_consumed
        FAME_L      = FAME / self.density_FAME
        energy_out  = FAME * self.LHV_FAME / 3600.0   # MW

        return {
            "FAME_kg_per_h":               float(FAME),
            "FAME_L_per_h":                float(FAME_L),
            "glycerol_kg_per_h":           float(glycerol),
            "methanol_consumed_kg_per_h":  float(MeOH),
            "energy_output_MW":            float(energy_out),
            "FAME_yield":                  float(self.FAME_yield),
        }
