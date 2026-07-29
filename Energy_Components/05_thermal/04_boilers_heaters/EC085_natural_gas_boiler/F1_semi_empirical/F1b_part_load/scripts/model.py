"""
EC085 — Natural Gas Boiler — F1b Part-Load with Flue & Standby Losses

Extends F1a with:
  1. Quadratic part-load curve: eta(PLR) = a0 + a1*PLR + a2*PLR^2
     with a0=0.75, a1=0.45, a2=-0.25 (peak eta ~0.95 at PLR ~0.9)
  2. Flue gas loss model: Q_flue = m_flue * cp_flue * (T_flue - T_air)
     T_flue scales with PLR (lower load = cooler flue)
  3. Standby loss: Q_standby = standby_fraction * Q_rated (casing/pilot)

The a0+a1+a2 = 0.95 != 1.0 in this model because efficiency is directly
computed from the polynomial (not multiplied by eta_nom).

References:
    EnergyPlus Engineering Reference (2023), Boiler:HotWater.
    EN 15502 — Gas-fired heating boilers.
    Stafford, A. (2009), Energy and Buildings, 41(2), 168-175.
"""

import numpy as np


class NaturalGasBoilerF1b:
    """Gas boiler with part-load curve, flue gas losses, and standby losses."""

    def __init__(self, params: dict):
        self.Q_rated = float(params["Q_rated"])
        self.a0 = float(params["a0"])
        self.a1 = float(params["a1"])
        self.a2 = float(params["a2"])
        self.PLR_min = float(params["PLR_min"])
        self.LHV_gas = float(params["LHV_gas"])  # MJ/m3
        self.T_flue_full = float(params["flue_gas_temp_full"])
        self.T_air = float(params["combustion_air_temp"])
        self.cp_flue = float(params["flue_gas_cp"])           # kJ/kgK
        self.excess_air = float(params["excess_air_ratio"])
        self.stoich_afr = float(params["stoich_air_fuel_ratio"])
        self.standby_frac = float(params["standby_loss_fraction"])

    # ------------------------------------------------------------------
    # Part-load efficiency
    # ------------------------------------------------------------------

    def efficiency(self, PLR):
        """
        eta(PLR) = a0 + a1*PLR + a2*PLR^2

        Peak at PLR_peak = -a1/(2*a2) = 0.9 for default coefficients.
        """
        PLR = np.asarray(PLR, dtype=float)
        PLR_eff = np.maximum(PLR, self.PLR_min)
        eta = self.a0 + self.a1 * PLR_eff + self.a2 * PLR_eff ** 2
        return np.clip(eta, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Thermal output and fuel input
    # ------------------------------------------------------------------

    def heat_output_kw(self, PLR):
        PLR = np.asarray(PLR, dtype=float)
        PLR_eff = np.maximum(PLR, self.PLR_min)
        return PLR_eff * self.Q_rated

    def fuel_input_kw(self, PLR):
        """Thermal power input from gas: Q_fuel = Q_out / eta(PLR)."""
        Q_out = self.heat_output_kw(PLR)
        eta = self.efficiency(PLR)
        safe_eta = np.where(eta > 0.01, eta, 0.01)
        return Q_out / safe_eta

    # ------------------------------------------------------------------
    # Flue gas loss
    # ------------------------------------------------------------------

    def flue_gas_temp(self, PLR):
        """
        Flue gas temperature scales with PLR.
        At full load: T_flue_full. At low load: drops linearly.
        T_flue(PLR) = T_air + (T_flue_full - T_air) * (0.3 + 0.7*PLR)
        """
        PLR = np.asarray(PLR, dtype=float)
        PLR_eff = np.maximum(PLR, self.PLR_min)
        return self.T_air + (self.T_flue_full - self.T_air) * (0.3 + 0.7 * PLR_eff)

    def flue_loss_kw(self, PLR, flue_gas_temp=None):
        """
        Flue gas sensible heat loss [kW].

        Q_flue = m_flue * cp_flue * (T_flue - T_air)
        m_flue ~ m_fuel * (1 + lambda * AFR_stoich)
        m_fuel ~ Q_fuel / LHV
        """
        PLR = np.asarray(PLR, dtype=float)
        Q_fuel = self.fuel_input_kw(PLR)

        # Mass flow of fuel (kg/s): Q_fuel_kW / (LHV_MJ/m3 * 1000/3600 * rho_gas)
        # Simplify: m_fuel_approx = Q_fuel / (LHV * 1000) where LHV in kJ/kg
        # For natural gas: LHV ~ 50 MJ/kg => m_fuel = Q_fuel / 50000
        m_fuel = Q_fuel / 50000.0  # kg/s (approximate)
        m_flue = m_fuel * (1.0 + self.excess_air * self.stoich_afr)

        if flue_gas_temp is None:
            T_flue = self.flue_gas_temp(PLR)
        else:
            T_flue = np.asarray(flue_gas_temp, dtype=float)

        # cp_flue in kJ/kgK, result in kW
        return m_flue * self.cp_flue * (T_flue - self.T_air)

    # ------------------------------------------------------------------
    # Standby loss
    # ------------------------------------------------------------------

    def standby_loss_kw(self):
        """Standby heat loss [kW] — casing loss, pilot flame."""
        return self.standby_frac * self.Q_rated

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(self, PLR, flue_gas_temp=None):
        """
        Full operating point evaluation.

        Parameters
        ----------
        PLR : float or array — part-load ratio [0, 1]
        flue_gas_temp : float or None — override flue temp (degC)

        Returns
        -------
        dict with efficiency, heat_output_kw, fuel_input_kw,
             flue_loss_kw, standby_loss_kw
        """
        PLR = np.asarray(PLR, dtype=float)
        return {
            "efficiency": self.efficiency(PLR),
            "heat_output_kw": self.heat_output_kw(PLR),
            "fuel_input_kw": self.fuel_input_kw(PLR),
            "flue_loss_kw": self.flue_loss_kw(PLR, flue_gas_temp),
            "standby_loss_kw": np.full_like(PLR, self.standby_loss_kw()),
        }
