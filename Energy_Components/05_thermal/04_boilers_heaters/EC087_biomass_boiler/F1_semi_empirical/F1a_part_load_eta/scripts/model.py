"""
EC087 — Biomass Boiler — F1a Part-Load Efficiency
Physics equations class.

Model:
  eta(PLR) = eta_nom * (a0 + a1*PLR + a2*PLR^2)   with a0+a1+a2 = 1
  Q_out    = PLR * Q_rated                         [kW thermal output]
  Q_fuel   = Q_out / eta(PLR)                      [kW thermal input]
  m_fuel   = Q_fuel / LHV_eff                      [kg/h]

The part-load curve is more aggressive than for natural gas because
biomass combustion suffers from incomplete combustion at low PLR
(unburnt carbon, CO emissions, temperature drops in firebox).

Effective LHV is corrected for moisture content (latent heat of vaporisation
of bound water reduces useful heat release):
    LHV_eff = LHV_dry * (1 - w) - 2.443 * w        [MJ/kg, w = moist mass frac]

Source:
    EN 303-5:2012 'Heating boilers for solid fuels';
    IEA Bioenergy Task 32 'Combustion of solid biomass';
    Carvalho et al. (2013) Energy 58, 290-301.
"""

import numpy as np


class BiomassBoilerModel:
    """
    Semi-empirical part-load efficiency model for an automatic
    wood pellet / wood chip biomass boiler (EN 303-5 class 5).
    """

    H_VAP_WATER = 2.443  # MJ/kg latent heat of vaporisation at 25°C

    def __init__(self, params: dict):
        self.Q_rated   = float(params["Q_rated"])
        self.eta_nom   = float(params["eta_nom"])
        self.a0        = float(params["a0"])
        self.a1        = float(params["a1"])
        self.a2        = float(params["a2"])
        self.PLR_min   = float(params["PLR_min"])
        self.LHV_dry   = float(params["LHV_fuel_MJ_kg"])
        self.moist     = float(params.get("moisture_content", 0.10))
        self.co2_fac   = float(params.get("co2_factor_g_per_kwh_th", 18.0))

        coeff_sum = self.a0 + self.a1 + self.a2
        if abs(coeff_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Part-load polynomial coefficients must sum to 1.0 "
                f"(got {coeff_sum:.6f})."
            )
        if not (0.0 <= self.moist < 1.0):
            raise ValueError(
                f"moisture_content must be in [0, 1), got {self.moist}"
            )

    # ------------------------------------------------------------------

    def effective_lhv_MJ_kg(self) -> float:
        """LHV corrected for fuel moisture (as-received basis)."""
        return self.LHV_dry * (1.0 - self.moist) - self.H_VAP_WATER * self.moist

    # ------------------------------------------------------------------

    def efficiency(self, PLR: float) -> float:
        """
        Part-load thermal efficiency [-].
        eta(PLR) = eta_nom * (a0 + a1*PLR + a2*PLR^2), clipped to [0,1].
        Below PLR_min the boiler cycles; we clamp PLR to PLR_min for the
        instantaneous-on efficiency value.
        """
        if PLR < self.PLR_min:
            PLR = self.PLR_min
        eta = self.eta_nom * (self.a0 + self.a1 * PLR + self.a2 * PLR ** 2)
        return float(np.clip(eta, 0.0, 1.0))

    def thermal_output_kw(self, PLR: float) -> float:
        """Useful thermal output [kW]."""
        PLR = max(PLR, self.PLR_min)
        return PLR * self.Q_rated

    def fuel_input_kw(self, PLR: float) -> float:
        """Thermal input from biomass combustion [kW]."""
        Q_out = self.thermal_output_kw(PLR)
        eta = self.efficiency(PLR)
        if eta <= 0.0:
            return float("inf")
        return Q_out / eta

    def fuel_mass_flow_kg_h(self, PLR: float) -> float:
        """Mass flow of as-received biomass fuel [kg/h]."""
        Q_fuel_kw = self.fuel_input_kw(PLR)
        LHV_eff = self.effective_lhv_MJ_kg()  # MJ/kg
        if LHV_eff <= 0.0:
            return float("inf")
        # kW = kJ/s; kJ/s * 3600 s/h / (1000 kJ/MJ) -> MJ/h
        # MJ/h / (MJ/kg) = kg/h
        return Q_fuel_kw * 3600.0 / 1000.0 / LHV_eff

    # ------------------------------------------------------------------

    def evaluate(self, PLR: float) -> dict:
        """Return all outputs for a given part-load ratio."""
        if PLR < 0 or PLR > 1.0:
            raise ValueError(f"PLR must be in [0, 1], got {PLR}")

        PLR_eff = max(PLR, self.PLR_min)
        eta = self.efficiency(PLR_eff)
        Q_out = PLR_eff * self.Q_rated
        Q_fuel = Q_out / eta if eta > 0 else float("inf")
        m_fuel = self.fuel_mass_flow_kg_h(PLR_eff)

        return {
            "PLR":                       PLR,
            "PLR_effective":             PLR_eff,
            "efficiency":                eta,
            "thermal_output_kw":         Q_out,
            "fuel_input_kw":             Q_fuel,
            "fuel_mass_flow_kg_h":       m_fuel,
            "LHV_effective_MJ_kg":       self.effective_lhv_MJ_kg(),
            "co2_emissions_g_per_kwh_th": self.co2_fac,
        }
