"""
EC086 — Electric Boiler / Resistance Heater — F1a Constant Efficiency
Physics equations class.

Model:
  Q_out = eta_nom * (P_elec - P_standby)            [kW]
  P_in  = PLR * P_rated + P_standby                  [kW]
  eta(PLR) = Q_out / P_in   ~ eta_nom for PLR > ~0.05

Resistance heaters convert electricity to heat with ~unity efficiency
(typically 0.99 accounting for small jacket / control losses).
There is no combustion, no flue, no fuel — therefore no part-load
combustion penalty and no CO2 emissions at point of use.

Source:
  ASHRAE Handbook (HVAC Systems & Equipment, 2020), Ch. 32 'Boilers';
  IEA Task 44 'Solar and Heat Pump Systems' reference electric heater.
"""

import numpy as np


class ElectricBoilerModel:
    """
    Constant-efficiency electric resistance boiler / heater.

    Energy balance
    --------------
        P_in  = PLR * P_rated + P_standby                [kW electrical]
        Q_out = eta_nom * P_in                           [kW thermal]
    The instantaneous efficiency is essentially eta_nom for any non-trivial
    load. Standby parasitics are explicitly tracked.
    """

    def __init__(self, params: dict):
        self.P_rated = float(params["P_rated_kw"])
        self.eta_nom = float(params["eta_nom"])
        self.P_standby = float(params.get("P_standby_kw", 0.0))
        self.PLR_min = float(params.get("PLR_min", 0.0))

        if not (0.0 < self.eta_nom <= 1.0):
            raise ValueError(
                f"eta_nom must be in (0, 1], got {self.eta_nom}"
            )
        if self.P_rated <= 0:
            raise ValueError(f"P_rated_kw must be > 0, got {self.P_rated}")

    # ------------------------------------------------------------------

    def electrical_input_kw(self, PLR: float) -> float:
        """Electrical input power [kW]."""
        PLR = max(PLR, self.PLR_min)
        return PLR * self.P_rated + self.P_standby

    def thermal_output_kw(self, PLR: float) -> float:
        """Thermal output power [kW]."""
        return self.eta_nom * self.electrical_input_kw(PLR)

    def efficiency(self, PLR: float) -> float:
        """Effective conversion efficiency [-] (Q_out / P_in)."""
        P_in = self.electrical_input_kw(PLR)
        if P_in <= 0.0:
            return 0.0
        return float(np.clip(self.thermal_output_kw(PLR) / P_in, 0.0, 1.0))

    # ------------------------------------------------------------------

    def evaluate(self, PLR: float) -> dict:
        """
        Return all outputs for a given part-load ratio.

        Parameters
        ----------
        PLR : float
            Part-load ratio in [0, 1].
        """
        if PLR < 0 or PLR > 1.0:
            raise ValueError(f"PLR must be in [0, 1], got {PLR}")

        PLR_eff = max(PLR, self.PLR_min)
        P_in = self.electrical_input_kw(PLR_eff)
        Q_out = self.thermal_output_kw(PLR_eff)
        eta = self.efficiency(PLR_eff)

        return {
            "PLR":                   PLR,
            "PLR_effective":         PLR_eff,
            "electrical_input_kw":   P_in,
            "thermal_output_kw":     Q_out,
            "efficiency":            eta,
            "co2_emissions_g_per_kwh_th": 0.0,  # zero at point of use
        }
