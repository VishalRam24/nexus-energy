"""
EC146 -- Torrefaction Reactor -- F1a Yield Model

Simplest semi-empirical model at fixed operating point (275 degC):
    torrefied_solid_kg = feedstock_dry_kg * solid_yield
    LHV_torrefied      = LHV_raw * energy_density_factor

Energy yield (fraction of feedstock energy retained in solid):
    energy_out = energy_in * energy_yield
    (energy_yield = solid_yield * energy_density_factor / 1.0 ~ 0.91)

F1b adds temperature-dependent yields, moisture effects.

References:
    Bergman, P.C.A. et al. (2005). ECN-C-05-073.
    van der Stelt, M.J.C. et al. (2011). Biomass Bioenergy 35:3748.
"""

import numpy as np


class TorrefactionReactorF1a:
    """Torrefaction reactor: torrefied_solid = feedstock * solid_yield."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.solid_yield          = float(u["solid_yield"]["value"])
        self.energy_yield         = float(u["energy_yield"]["value"])
        self.energy_density_factor = float(u["energy_density_factor"]["value"])
        self.T_operating          = float(u["T_operating_degC"]["value"])
        self.LHV_raw              = float(u["LHV_raw_MJ_kg"]["value"])
        self.LHV_torrefied        = float(u["LHV_torrefied_MJ_kg"]["value"])
        self.volatile_loss        = float(u["volatile_loss_fraction"]["value"])

    def predict(self, feedstock_dry_kg_per_h: float) -> dict:
        """
        Parameters
        ----------
        feedstock_dry_kg_per_h : float  Dry biomass feed rate [kg/h]

        Returns
        -------
        dict with:
            torrefied_solid_kg_per_h  : solid product [kg/h]
            volatile_loss_kg_per_h    : volatiles (torgas) [kg/h]
            energy_in_MW              : energy in feedstock [MW]
            energy_out_MW             : energy in torrefied solid [MW]
            energy_yield              : fraction of energy retained [-]
            LHV_torrefied_MJ_kg       : energy density of product [MJ/kg]
            energy_density_factor     : densification ratio [-]
        """
        feed = float(np.clip(feedstock_dry_kg_per_h, 0.0, None))

        solid   = feed * self.solid_yield
        volatiles = feed * self.volatile_loss

        energy_in  = feed  * self.LHV_raw      / 3600.0   # MW
        energy_out = solid * self.LHV_torrefied / 3600.0   # MW

        return {
            "torrefied_solid_kg_per_h": float(solid),
            "volatile_loss_kg_per_h":   float(volatiles),
            "energy_in_MW":             float(energy_in),
            "energy_out_MW":            float(energy_out),
            "energy_yield":             float(self.energy_yield),
            "LHV_torrefied_MJ_kg":      float(self.LHV_torrefied),
            "energy_density_factor":    float(self.energy_density_factor),
        }
