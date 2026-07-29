"""
EC094 — Evaporative Cooler — F1b COP vs Temperature and Part-Load

Extends F1a (constant EER/COP map) with:
  1. Saturation effectiveness model: eps_s = (T_wb,in - T_wb,out) / (T_wb,in - T_wb,in_dewpoint)
     For direct evaporative: eta_eff = (T_db,in - T_db,out) / (T_db,in - T_wb,in)
  2. COP variation with wet-bulb depression (T_db - T_wb):
     Larger wet-bulb depression → better evaporation potential → higher EER
  3. Part-load penalty:
     COP_pl(PLR) = COP_ref * (d1 + d2*PLR + d3*PLR^2)   [DOE-2 form]
     Evaporative coolers have disproportionate fan energy at part load.
  4. Ambient humidity penalty:
     High relative humidity reduces effectiveness → lower cooling output

Energy model:
    Q_cool = m_air * cp_air * (T_in - T_out)           [kW]
    T_out  = T_db,in - eps * (T_db,in - T_wb,in)       [direct evap]
    W_fan  = W_fan_rated * PLR^2.5                      [fan affinity law]
    EER    = Q_cool / W_fan                             [cooling / electrical]

References:
    ASHRAE Handbook Fundamentals (2021), Chapter 1 — Psychrometrics.
    Watt, J.R. & Brown, W.K. (1997). Evaporative Air Conditioning Handbook. Fairmont Press.
    AHRI Standard 210/240 — Performance Rating of Unitary Air-Conditioning Equipment.
    Cengel & Boles (2014). Thermodynamics: An Engineering Approach, 8e.
"""

import numpy as np


class EvaporativeCoolerF1b:
    """Direct evaporative cooler with saturation effectiveness, humidity correction, part-load."""

    def __init__(self, params: dict):
        self.Q_rated         = float(params["Q_rated_kw"])
        self.W_fan_rated     = float(params["W_fan_rated_kw"])
        self.eps_design      = float(params["saturation_effectiveness"])  # 0.7-0.9 typical
        self.PLR_min         = float(params["PLR_min"])
        # DOE-2 part-load EER curve
        self.d1 = float(params["d1"])
        self.d2 = float(params["d2"])
        self.d3 = float(params["d3"])
        # Humidity penalty: EER_adj = EER * max(1 - k_rh*(RH - RH_design), 0.3)
        self.k_rh          = float(params["k_rh"])     # penalty per %RH above design
        self.RH_design      = float(params["RH_design"])  # %

    # ------------------------------------------------------------------
    # Psychrometrics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def wet_bulb_approx(T_db, RH):
        """
        Approximate wet-bulb temperature [degC] from dry-bulb and RH [%].
        Stull (2011) empirical formula: valid for T in [-20, 50], RH in [5%, 100%].
        T_wb = T_db * atan(0.151977*(RH+8.313659)^0.5) + atan(T_db+RH)
               - atan(RH-1.676331) + 0.00391838*RH^1.5*atan(0.023101*RH)
               - 4.686035
        Simplified version for engineering use.
        """
        T  = np.asarray(T_db, dtype=float)
        rh = np.asarray(RH,   dtype=float)
        T_wb = (T * np.arctan(0.151977 * (rh + 8.313659) ** 0.5)
                + np.arctan(T + rh)
                - np.arctan(rh - 1.676331)
                + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
                - 4.686035)
        return T_wb

    # ------------------------------------------------------------------
    # Outlet temperature
    # ------------------------------------------------------------------

    def outlet_temp(self, T_db_in, T_wb_in):
        """
        Direct evaporative cooling outlet temperature.
        T_out = T_db,in - eps * (T_db,in - T_wb,in)
        """
        T_db = np.asarray(T_db_in, dtype=float)
        T_wb = np.asarray(T_wb_in, dtype=float)
        return T_db - self.eps_design * (T_db - T_wb)

    # ------------------------------------------------------------------
    # Cooling capacity
    # ------------------------------------------------------------------

    def cooling_power_kw(self, T_db_in, RH, m_air_kg_s=None):
        """
        Cooling power [kW].
        Q = m_air * cp_air * (T_db,in - T_out)
        Default m_air from rated conditions if not provided.
        """
        T_db = np.asarray(T_db_in, dtype=float)
        rh   = np.asarray(RH, dtype=float)
        T_wb = self.wet_bulb_approx(T_db, rh)
        T_out = self.outlet_temp(T_db, T_wb)

        cp_air = 1.005  # kJ/(kg·K)
        # Rated mass flow at design conditions (back-calculate)
        if m_air_kg_s is None:
            dT_design = T_db - T_wb   # use current conditions
            dT_design = np.where(dT_design > 0.5, dT_design, 0.5)
            Q_approx  = self.eps_design * dT_design
            m_air_ref = self.Q_rated / (cp_air * np.where(Q_approx > 0.1, Q_approx, 0.1))
            m_air     = m_air_ref
        else:
            m_air = np.asarray(m_air_kg_s, dtype=float)

        dT = T_db - T_out
        return np.maximum(m_air * cp_air * dT, 0.0)

    # ------------------------------------------------------------------
    # EER with part-load and humidity
    # ------------------------------------------------------------------

    def eer_f_plr(self, plr):
        """DOE-2 part-load EER correction."""
        plr_eff = np.maximum(np.asarray(plr, dtype=float), self.PLR_min)
        return self.d1 + self.d2 * plr_eff + self.d3 * plr_eff ** 2

    def humidity_correction(self, RH):
        """
        EER reduction at high humidity.
        f_RH = max(1 - k_rh*(RH - RH_design)/100, 0.3)
        """
        rh = np.asarray(RH, dtype=float)
        f  = 1.0 - self.k_rh * np.maximum(rh - self.RH_design, 0.0) / 100.0
        return np.clip(f, 0.3, 1.2)

    def eer(self, T_db_in, RH, plr=1.0):
        """
        Effective EER [kW_cool / kW_elec].
        EER = EER_rated * eer_f_plr(PLR) * f_RH(RH)
        """
        EER_rated = self.Q_rated / max(self.W_fan_rated, 0.001)
        plr_arr   = np.asarray(plr, dtype=float)
        f_plr     = self.eer_f_plr(plr_arr)
        f_rh      = self.humidity_correction(RH)
        return np.maximum(EER_rated * f_plr * f_rh, 0.1)

    # ------------------------------------------------------------------
    # Fan power
    # ------------------------------------------------------------------

    def fan_power_kw(self, plr=1.0):
        """
        Fan power using affinity law: W_fan = W_rated * PLR^2.5
        (Realistic: exponent between 2 and 3 depending on fan type.)
        """
        plr_eff = np.maximum(np.asarray(plr, dtype=float), self.PLR_min)
        return self.W_fan_rated * plr_eff ** 2.5

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(self, T_db_in, RH, plr=1.0):
        """
        Full evaluation at given conditions.

        Parameters
        ----------
        T_db_in : float or array — inlet dry-bulb temperature [degC]
        RH      : float or array — relative humidity [%]
        plr     : float or array — part-load ratio [0, 1]

        Returns
        -------
        dict with Q_cool_kw, W_fan_kw, EER, T_wb, T_outlet_c, eta_collector
        """
        T_db = np.asarray(T_db_in, dtype=float)
        rh   = np.asarray(RH,      dtype=float)
        plr  = np.asarray(plr,     dtype=float)

        T_wb   = self.wet_bulb_approx(T_db, rh)
        T_out  = self.outlet_temp(T_db, T_wb)
        W_fan  = self.fan_power_kw(plr)
        Q_cool = self.Q_rated * np.maximum(plr, self.PLR_min)
        # Scale Q_cool by humidity correction
        f_rh   = self.humidity_correction(rh)
        Q_cool = Q_cool * f_rh
        eer    = self.eer(T_db, rh, plr)

        return {
            "Q_cool_kw":   Q_cool,
            "W_fan_kw":    W_fan,
            "EER":         eer,
            "T_wb_c":      T_wb,
            "T_outlet_c":  T_out,
            "f_humidity":  f_rh,
            "f_partload":  self.eer_f_plr(plr),
        }
