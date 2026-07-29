"""
EC085 — Natural Gas Boiler — F1a Part-Load Efficiency
Physics equations class.

Model:
  eta(PLR) = eta_nom * (a0 + a1*PLR + a2*PLR^2)   where a0+a1+a2=1
  Q_out    = Q_rated * PLR * eta(PLR)
  fuel_input = Q_out / eta(PLR) / LHV              [alternative: Q_rated*PLR]
             = Q_rated * PLR / eta(PLR)             [thermal input basis]

Note: the fuel input (thermal) is:
  Q_fuel = Q_rated * PLR / eta(PLR)    [kW thermal input from gas combustion]
  Gas consumption: V_gas = Q_fuel / LHV_gas         [m³/h at operating conditions]

Source: EnergyPlus Engineering Reference (2023), Section "Boiler:HotWater";
        Stafford, A. (2009), "The performance of domestic condensing boilers",
        Energy and Buildings 41(2), 168-175.
"""

import numpy as np


class NaturalGasBoilerModel:
    """
    Semi-empirical part-load efficiency model for a natural gas condensing boiler.

    Part-load efficiency
    --------------------
    eta(PLR) = eta_nom * (a0 + a1*PLR + a2*PLR^2)

    The polynomial coefficients (a0, a1, a2) satisfy a0+a1+a2 = 1 so that
    eta(PLR=1) = eta_nom exactly.

    For a nearly linear boiler: a0=0.1, a1=0.9, a2=0.0.

    Thermal output
    --------------
    Q_out [kW] = Q_rated [kW] * PLR * eta(PLR)   — NOT correct.

    Correct formulation (EnergyPlus convention):
    The part-load curve modifies the efficiency.  The boiler thermal output is:
        Q_out = PLR * Q_rated                    [user-requested heat]
    and the fuel consumed is:
        Q_fuel = Q_out / eta(PLR)                [thermal input from gas]

    We follow this convention:
        Q_out_kW  = PLR * Q_rated_kW
        Q_fuel_kW = Q_out_kW / eta(PLR)
        V_gas_m3h = Q_fuel_kW * 3600 / (LHV_gas_MJ_m3 * 1000)   [m³/h]
    """

    def __init__(self, params: dict):
        """
        Parameters
        ----------
        params : dict
            Q_rated         : rated thermal output [kW]
            eta_nom         : nominal (full-load) efficiency [-]
            a0, a1, a2      : part-load curve polynomial coefficients
            PLR_min         : minimum part-load ratio [-]
            LHV_gas         : lower heating value of natural gas [MJ/m³]
        """
        self.Q_rated  = float(params["Q_rated"])
        self.eta_nom  = float(params["eta_nom"])
        self.a0       = float(params["a0"])
        self.a1       = float(params["a1"])
        self.a2       = float(params["a2"])
        self.PLR_min  = float(params["PLR_min"])
        self.LHV_gas  = float(params["LHV_gas"])   # MJ/m³

        # Validate polynomial constraint
        coeff_sum = self.a0 + self.a1 + self.a2
        if abs(coeff_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Part-load polynomial coefficients must sum to 1.0 "
                f"(got {coeff_sum:.6f}). Adjust a0, a1, a2."
            )

    # ------------------------------------------------------------------
    # Efficiency model
    # ------------------------------------------------------------------

    def efficiency(self, PLR: float) -> float:
        """
        Part-load efficiency [-].
        eta(PLR) = eta_nom * (a0 + a1*PLR + a2*PLR^2)
        Clipped to [0, 1].
        """
        if PLR < self.PLR_min:
            PLR = self.PLR_min   # boiler modulates to minimum
        eta = self.eta_nom * (self.a0 + self.a1 * PLR + self.a2 * PLR ** 2)
        return float(np.clip(eta, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Thermal and fuel quantities
    # ------------------------------------------------------------------

    def thermal_output_kw(self, PLR: float) -> float:
        """
        Useful thermal output [kW].
        Q_out = PLR * Q_rated
        """
        PLR = max(PLR, self.PLR_min)
        return PLR * self.Q_rated

    def fuel_input_kw(self, PLR: float) -> float:
        """
        Thermal power input from gas combustion [kW].
        Q_fuel = Q_out / eta(PLR)
        """
        Q_out = self.thermal_output_kw(PLR)
        eta   = self.efficiency(PLR)
        if eta <= 0.0:
            return float("inf")
        return Q_out / eta

    def gas_consumption_m3h(self, PLR: float) -> float:
        """
        Volumetric gas consumption [m³/h] at standard conditions.
        V_gas = Q_fuel [kW] * 3600 [s/h] / (LHV [MJ/m³] * 1e6 [J/MJ] / 1e3 [W/kW])
              = Q_fuel_kW * 3600 / (LHV_MJ_m3 * 1000)
        """
        Q_fuel = self.fuel_input_kw(PLR)
        # LHV in MJ/m³ -> convert to kWh/m³ = MJ/m³ / 3.6
        # V = Q_fuel_kW / (LHV_kWh_m3) = Q_fuel_kW / (LHV_MJ_m3 / 3.6)
        LHV_kWh_m3 = self.LHV_gas / 3.6
        return Q_fuel / LHV_kWh_m3

    # ------------------------------------------------------------------
    # Temperature correction (supply temperature effect on condensing eta)
    # ------------------------------------------------------------------

    def condensing_correction(self, T_supply_C: float) -> float:
        """
        Approximate correction factor for condensing boiler efficiency
        as a function of supply water temperature.

        At low supply temperatures (< 55°C), the flue gas condenses and
        efficiency increases.  Linear interpolation between:
          T=30°C  -> factor=1.05 (significant condensing benefit)
          T=55°C  -> factor=1.00 (onset of condensing)
          T=80°C  -> factor=0.92 (non-condensing regime, some penalty)

        This factor multiplies the eta_nom in the efficiency calculation.
        """
        if T_supply_C <= 55.0:
            # Linear: 1.05 at 30°C, 1.00 at 55°C
            factor = 1.05 - (T_supply_C - 30.0) * (0.05 / 25.0)
        else:
            # Linear: 1.00 at 55°C, 0.92 at 80°C
            factor = 1.00 - (T_supply_C - 55.0) * (0.08 / 25.0)
        return float(np.clip(factor, 0.80, 1.10))

    def efficiency_with_temp(self, PLR: float, T_supply_C: float) -> float:
        """Efficiency accounting for supply temperature correction."""
        base_eta = self.efficiency(PLR)
        correction = self.condensing_correction(T_supply_C)
        return float(np.clip(base_eta * correction, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Full operating-point evaluation
    # ------------------------------------------------------------------

    def evaluate(self, PLR: float, T_supply_C: float = 60.0) -> dict:
        """
        Return all outputs for a given part-load ratio and supply temperature.

        Parameters
        ----------
        PLR         : part-load ratio [0, 1]
        T_supply_C  : supply water temperature [°C]
        """
        if PLR < 0 or PLR > 1.0:
            raise ValueError(f"PLR must be in [0, 1], got {PLR}")
        if T_supply_C < 30.0 or T_supply_C > 80.0:
            raise ValueError(f"T_supply must be in [30, 80] °C, got {T_supply_C}")

        PLR_eff = max(PLR, self.PLR_min)
        eta_base = self.efficiency(PLR_eff)
        corr     = self.condensing_correction(T_supply_C)
        eta_eff  = float(np.clip(eta_base * corr, 0.0, 1.0))

        Q_out    = PLR_eff * self.Q_rated
        Q_fuel   = Q_out / eta_eff if eta_eff > 0 else float("inf")
        V_gas    = Q_fuel / (self.LHV_gas / 3.6)

        return {
            "PLR":                  PLR,
            "PLR_effective":        PLR_eff,
            "T_supply_C":           T_supply_C,
            "efficiency_base":      eta_base,
            "condensing_factor":    corr,
            "efficiency":           eta_eff,
            "thermal_output_kw":    Q_out,
            "fuel_input_kw":        Q_fuel,
            "gas_consumption_m3h":  V_gas,
        }
