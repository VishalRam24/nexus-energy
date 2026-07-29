"""
EC088 — Oil-Fired Boiler — F1b Part-Load Efficiency with Flue & Standby Losses

Extends F1a (constant efficiency) with:
  1. Quadratic part-load curve: eta(PLR) = a0 + a1*PLR + a2*PLR^2
     Typical oil boiler peak ~0.89 at PLR~0.85 (slightly lower than NG due to viscosity)
  2. Flue gas sensible-heat loss:
     Q_flue = m_flue * cp_flue * (T_flue - T_air)
     T_flue(PLR) = T_air + (T_flue_full - T_air) * (0.32 + 0.68*PLR)
     Oil has higher flue temperature due to sulfur content and atomization characteristics.
  3. Standby/casing loss:
     Q_standby = standby_frac * Q_rated

Oil-specific considerations:
  - Stoichiometric AFR ≈ 13.8 (vs 17.2 for natural gas) — heavier fuel
  - LHV_oil ≈ 42 MJ/kg (No.2 fuel oil / light heating oil)
  - Excess air ratio typically 0.15-0.25 (λ = 1.15–1.25) for oil burners

References:
    EN 303-1:2017 — Heating boilers — Gas-fired central heating boilers.
    ASHRAE Handbook HVAC Systems & Equipment (2020), Ch. 32.
    Buderus (2022) — Oil boiler technical specification GB125 series.
    EnergyPlus Engineering Reference (2023), Boiler:HotWater.
"""

import numpy as np


class OilBoilerF1b:
    """Oil-fired boiler with part-load curve, flue gas, and standby losses."""

    def __init__(self, params: dict):
        self.Q_rated       = float(params["Q_rated"])
        self.a0            = float(params["a0"])
        self.a1            = float(params["a1"])
        self.a2            = float(params["a2"])
        self.PLR_min       = float(params["PLR_min"])
        self.LHV_oil_mj_kg = float(params["LHV_oil_mj_kg"])    # MJ/kg
        self.T_flue_full   = float(params["T_flue_full"])       # degC
        self.T_air         = float(params["T_air"])             # degC
        self.cp_flue       = float(params["cp_flue_kj_kgk"])    # kJ/(kg·K)
        self.excess_air    = float(params["excess_air_ratio"])
        self.stoich_afr    = float(params["stoich_afr"])        # ~13.8 for fuel oil
        self.standby_frac  = float(params["standby_loss_fraction"])

    # ------------------------------------------------------------------
    # Part-load efficiency
    # ------------------------------------------------------------------

    def efficiency(self, PLR):
        """eta(PLR) = a0 + a1*PLR + a2*PLR^2"""
        PLR     = np.asarray(PLR, dtype=float)
        PLR_eff = np.maximum(PLR, self.PLR_min)
        eta     = self.a0 + self.a1 * PLR_eff + self.a2 * PLR_eff ** 2
        return np.clip(eta, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Thermal output and fuel
    # ------------------------------------------------------------------

    def heat_output_kw(self, PLR):
        PLR_eff = np.maximum(np.asarray(PLR, dtype=float), self.PLR_min)
        return PLR_eff * self.Q_rated

    def fuel_input_kw(self, PLR):
        """Q_fuel = Q_out / eta  [kW, LHV basis]"""
        Q_out   = self.heat_output_kw(PLR)
        eta     = self.efficiency(PLR)
        safe    = np.where(eta > 0.01, eta, 0.01)
        return Q_out / safe

    # ------------------------------------------------------------------
    # Flue gas loss
    # ------------------------------------------------------------------

    def flue_gas_temp(self, PLR):
        """
        T_flue(PLR) = T_air + (T_flue_full - T_air) * (0.32 + 0.68*PLR)
        Oil boilers maintain higher minimum flue temperature to avoid sulfur condensation.
        """
        PLR_eff = np.maximum(np.asarray(PLR, dtype=float), self.PLR_min)
        return self.T_air + (self.T_flue_full - self.T_air) * (0.32 + 0.68 * PLR_eff)

    def flue_loss_kw(self, PLR, flue_gas_temp=None):
        """
        Q_flue = m_flue * cp_flue * (T_flue - T_air)
        m_fuel = Q_fuel / LHV_oil  [kg/s]
        m_flue = m_fuel * (1 + lambda * AFR_stoich)  [kg/s]
        """
        PLR    = np.asarray(PLR, dtype=float)
        Q_fuel = self.fuel_input_kw(PLR)

        # Fuel mass flow: kW / (kJ/kg) = kg/s
        LHV_kj_kg = self.LHV_oil_mj_kg * 1000.0
        m_fuel     = Q_fuel / LHV_kj_kg
        m_flue     = m_fuel * (1.0 + self.excess_air * self.stoich_afr)

        T_flue = self.flue_gas_temp(PLR) if flue_gas_temp is None \
                 else np.asarray(flue_gas_temp, dtype=float)

        return m_flue * self.cp_flue * (T_flue - self.T_air)

    # ------------------------------------------------------------------
    # Standby loss
    # ------------------------------------------------------------------

    def standby_loss_kw(self):
        return self.standby_frac * self.Q_rated

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(self, PLR, flue_gas_temp=None):
        PLR = np.asarray(PLR, dtype=float)
        return {
            "efficiency":       self.efficiency(PLR),
            "heat_output_kw":   self.heat_output_kw(PLR),
            "fuel_input_kw":    self.fuel_input_kw(PLR),
            "flue_loss_kw":     self.flue_loss_kw(PLR, flue_gas_temp),
            "standby_loss_kw":  np.full_like(PLR, self.standby_loss_kw()),
            "flue_gas_temp_c":  self.flue_gas_temp(PLR),
        }
