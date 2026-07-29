"""
EC152 — Flash Steam Geothermal Plant — F1a Exergy Efficiency Model

Flash steam plants reduce pressure of hot brine causing partial flashing to steam,
which is separated and fed to turbines. Single or double flash configurations.

Optimal flash temperature for single flash (DiPippo, 2015):
    T_flash_opt = (T_geo_K * T_cond_K)**0.5 - 273.15   [geometric mean]

Steam quality from flash (approx., using specific enthalpy of vaporization):
    x = cp_brine * (T_geo - T_flash) / h_fg(T_flash)

Overall model:
    eta_Carnot = 1 - T_cond / T_geo   (K)
    eta_plant  = eta_util * eta_Carnot
    P_net      = m_dot_brine * cp_brine * (T_geo - T_reinject) * eta_plant

h_fg is approximated via Antoine/steam tables polynomial valid for 100-250°C:
    h_fg(T_c) ≈ 2501 - 2.361*T_c  kJ/kg  (linear fit to steam tables)

Reference:
    DiPippo, R. (2015). Geothermal Power Plants, 4th ed. Butterworth-Heinemann.
    Chapters 5–6 — Single-Flash and Double-Flash Steam Plants.
"""

import numpy as np


class FlashSteamGeothermalF1a:
    """
    Flash steam geothermal power plant — exergy-based efficiency model
    with optimal flash temperature calculation.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.cp_brine = u["cp_brine"]["value"]                      # J/(kg·K)
        self.eta_util = u["eta_utilization"]["value"]               # dimensionless
        self.T_condenser_offset = u["T_condenser_offset"]["value"]  # degC

    # ------------------------------------------------------------------
    # Steam table helpers
    # ------------------------------------------------------------------
    @staticmethod
    def h_fg(T_c):
        """
        Latent heat of vaporization at temperature T_c (°C) in kJ/kg.
        Linear fit to steam tables valid for 100–250°C.
        From Çengel & Boles (2014), Table A-4.
        """
        T = np.asarray(T_c, dtype=float)
        return np.clip(2501.0 - 2.361 * T, 100.0, None)  # kJ/kg

    # ------------------------------------------------------------------
    # Flash temperature
    # ------------------------------------------------------------------
    def optimal_flash_temperature(self, T_geo_c, T_reject_c):
        """
        Optimal flash (separator) temperature for single-flash plant.
        T_flash = sqrt(T_geo_K * T_cond_K) - 273.15  (geometric mean, degC)

        Parameters
        ----------
        T_geo_c    : geothermal brine temperature (degC)
        T_reject_c : cooling rejection temperature (degC)

        Returns
        -------
        T_flash_c : optimal flash temperature (degC)
        """
        T_geo  = np.asarray(T_geo_c, dtype=float) + 273.15
        T_cond = np.asarray(T_reject_c, dtype=float) + self.T_condenser_offset + 273.15
        T_flash_K = np.sqrt(T_geo * T_cond)
        return T_flash_K - 273.15

    # ------------------------------------------------------------------
    # Steam quality
    # ------------------------------------------------------------------
    def steam_quality(self, T_geo_c, T_flash_c):
        """
        Dryness fraction of steam produced in flash separator.
        x = cp * (T_geo - T_flash) / h_fg(T_flash)

        Returns
        -------
        x : steam quality [0–1]
        """
        T_geo   = np.asarray(T_geo_c,   dtype=float)
        T_flash = np.asarray(T_flash_c, dtype=float)
        dT = np.clip(T_geo - T_flash, 0.0, None)
        hfg = self.h_fg(T_flash) * 1000.0  # kJ/kg → J/kg
        x = self.cp_brine * dT / hfg
        return np.clip(x, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Thermodynamic efficiency
    # ------------------------------------------------------------------
    def carnot_efficiency(self, T_geo_c, T_reject_c):
        """
        Carnot efficiency between brine source and condenser.

        Parameters
        ----------
        T_geo_c    : brine wellhead temperature (degC)
        T_reject_c : cooling tower / air rejection temperature (degC)

        Returns
        -------
        eta_Carnot : ideal thermodynamic efficiency
        """
        T_geo  = np.asarray(T_geo_c, dtype=float) + 273.15
        T_cond = np.asarray(T_reject_c, dtype=float) + self.T_condenser_offset + 273.15
        eta = 1.0 - T_cond / T_geo
        return np.clip(eta, 0.0, 1.0)

    def plant_efficiency(self, T_geo_c, T_reject_c):
        """Overall plant efficiency = eta_util * eta_Carnot."""
        return self.eta_util * self.carnot_efficiency(T_geo_c, T_reject_c)

    # ------------------------------------------------------------------
    # Energy outputs
    # ------------------------------------------------------------------
    def condenser_temperature(self, T_reject_c):
        """Condenser saturation temperature (degC)."""
        return np.asarray(T_reject_c, dtype=float) + self.T_condenser_offset

    def heat_input(self, T_geo_c, T_reject_c, m_dot_kgs, T_flash_c=None):
        """
        Thermal energy available from brine before reinjection (kW).
        Q = m_dot * cp * (T_geo - T_reinject)

        T_reinject = T_flash (steam separated, liquid reinjected at flash T)
        If T_flash_c is None, uses the optimal flash temperature.
        """
        T_geo   = np.asarray(T_geo_c, dtype=float)
        m_dot   = np.asarray(m_dot_kgs, dtype=float)
        if T_flash_c is None:
            T_flash = self.optimal_flash_temperature(T_geo, T_reject_c)
        else:
            T_flash = np.asarray(T_flash_c, dtype=float)
        dT = np.clip(T_geo - T_flash, 0.0, None)
        return m_dot * self.cp_brine * dT / 1000.0  # W → kW

    def power_output(self, T_geo_c, T_reject_c, m_dot_kgs, T_flash_c=None):
        """
        Net electrical power output (kW).
        P = Q_heat * eta_plant
        """
        Q   = self.heat_input(T_geo_c, T_reject_c, m_dot_kgs, T_flash_c)
        eta = self.plant_efficiency(T_geo_c, T_reject_c)
        return Q * eta
