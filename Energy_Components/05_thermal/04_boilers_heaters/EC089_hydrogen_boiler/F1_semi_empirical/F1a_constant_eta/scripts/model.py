"""
EC089 — Hydrogen Boiler (100% H2 Combustion) — F1a Constant Efficiency
Physics equations class.

Model:
  Q_out  = PLR * Q_rated                              [kW thermal]
  Q_fuel = Q_out / eta_nom                            [kW thermal input]
  m_H2   = Q_fuel / LHV_H2                            [kg/h]

Hydrogen-only combustion produces water vapour as the only product:
    2 H2 + O2 -> 2 H2O
No CO2 is emitted at the point of use. NOx may form thermally but is
not modelled at F1a fidelity.

This model assumes a constant nominal efficiency typical of modern
condensing hydrogen-ready boilers (0.85-0.95). Part-load behaviour
is captured at higher fidelity (F1b/F1c).

Source:
    Hy4Heat WP6 (2021), 'Hydrogen-fuelled appliances safety case';
    BEIS UK Hydrogen Heating Trials (2022);
    Cellek & Pinarbasi (2018), Int. J. Hydrogen Energy 43, 1194-1207.
"""

import numpy as np


class HydrogenBoilerModel:
    """Constant-efficiency hydrogen boiler model."""

    def __init__(self, params: dict):
        self.Q_rated   = float(params["Q_rated"])
        self.eta_nom   = float(params["eta_nom"])
        self.PLR_min   = float(params.get("PLR_min", 0.1))
        self.LHV_H2    = float(params["LHV_H2_MJ_kg"])         # MJ/kg
        self.P_standby = float(params.get("P_standby_kw", 0.0))
        self.co2_fac   = float(params.get("co2_factor_g_per_kwh_th", 0.0))

        if not (0.0 < self.eta_nom <= 1.0):
            raise ValueError(f"eta_nom must be in (0, 1], got {self.eta_nom}")

    # ------------------------------------------------------------------

    def efficiency(self, PLR: float) -> float:
        """Constant nominal efficiency (PLR-independent at F1a)."""
        if PLR <= 0.0:
            return 0.0
        return float(np.clip(self.eta_nom, 0.0, 1.0))

    def thermal_output_kw(self, PLR: float) -> float:
        PLR = max(PLR, self.PLR_min)
        return PLR * self.Q_rated

    def fuel_input_kw(self, PLR: float) -> float:
        Q_out = self.thermal_output_kw(PLR)
        return Q_out / self.eta_nom

    def h2_mass_flow_kg_h(self, PLR: float) -> float:
        """Hydrogen mass flow [kg/h]."""
        Q_fuel_kw = self.fuel_input_kw(PLR)  # kW = kJ/s
        # MJ/h = Q_fuel_kw * 3600 / 1000
        # kg/h = MJ/h / LHV[MJ/kg]
        return Q_fuel_kw * 3.6 / self.LHV_H2

    def water_vapour_kg_h(self, PLR: float) -> float:
        """
        Water vapour produced from H2 combustion: 2 H2 + O2 -> 2 H2O.
        Mass ratio: 1 kg H2 -> 9.0 kg H2O (M_H2O / M_H2 = 18.015 / 2.016).
        """
        return self.h2_mass_flow_kg_h(PLR) * (18.015 / 2.016)

    # ------------------------------------------------------------------

    def evaluate(self, PLR: float) -> dict:
        if PLR < 0 or PLR > 1.0:
            raise ValueError(f"PLR must be in [0, 1], got {PLR}")

        PLR_eff = max(PLR, self.PLR_min)
        eta = self.efficiency(PLR_eff)
        Q_out = PLR_eff * self.Q_rated
        Q_fuel = Q_out / eta if eta > 0 else float("inf")
        m_h2 = self.h2_mass_flow_kg_h(PLR_eff)
        m_h2o = self.water_vapour_kg_h(PLR_eff)

        return {
            "PLR":                       PLR,
            "PLR_effective":             PLR_eff,
            "efficiency":                eta,
            "thermal_output_kw":         Q_out,
            "fuel_input_kw":             Q_fuel,
            "h2_mass_flow_kg_h":         m_h2,
            "water_vapour_kg_h":         m_h2o,
            "standby_power_kw":          self.P_standby,
            "co2_emissions_g_per_kwh_th": self.co2_fac,
        }
