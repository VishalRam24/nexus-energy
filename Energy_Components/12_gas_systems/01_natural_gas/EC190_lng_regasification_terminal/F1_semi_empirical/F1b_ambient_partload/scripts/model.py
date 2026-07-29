"""
EC190 — LNG Regasification Terminal — F1b Ambient + Part-Load Model

Extends F1a energy model with:
  1. Ambient temperature correction to SEC:
       SEC(T_amb) = SEC_base + sec_T_coeff * (T_amb - T_amb_ref)
     Warmer ambient (seawater) reduces fuel/electricity need for vaporisation.
  2. Part-load SEC penalty:
       SEC_PLR = SEC * (a + b * PLR)   at PLR=1 → factor=1.0, lower PLR → higher factor
  3. Combined net SEC with cold energy recovery credit.
  4. Total sendout capacity as fraction of design capacity.

References:
    Mokhatab, S. et al. (2014). Handbook of Liquefied Natural Gas. Elsevier.
    Shah, N. et al. (2013). J. Natural Gas Science & Engineering, 11, 9-16.
    DNV GL (2018). LNG Plant Technology — Operational Efficiency.
"""

import numpy as np


class LNGRegasF1b:
    """LNG regasification terminal with ambient temperature and part-load corrections."""

    def __init__(self, params: dict):
        t = params["terminal"]
        g = params["gas"]

        self.sec_base = t["sec_base"]["value"]          # kWh/ton
        self.f_cold = t["cold_energy_recovery_fraction"]["value"]
        self.h_fg = t["vaporisation_heat"]["value"]     # kJ/kg
        self.PLR_coeffs = t["PLR_coeffs"]["value"]      # [a, b]
        self.T_amb_ref = t["T_amb_ref"]["value"]        # K
        self.sec_T_coeff = t["sec_T_coeff"]["value"]    # kWh/(ton.K)

        self.T_storage = g["T_storage"]["value"]        # K (LNG storage ~-162°C)
        self.T_sendout = g["T_sendout"]["value"]        # K

    def sec_ambient_corrected(self, T_ambient_K):
        """
        SEC corrected for ambient temperature [kWh/ton].
        Warmer seawater → less fuel/electricity for vaporisation.
        """
        T = np.asarray(T_ambient_K, dtype=float)
        sec = self.sec_base + self.sec_T_coeff * (T - self.T_amb_ref)
        return np.clip(sec, 10.0, 150.0)

    def plr_factor(self, plr):
        """
        Part-load SEC multiplier.
        factor = a + b * PLR → at PLR=1: 1.0; lower PLR → higher factor (penalty).
        """
        plr = np.asarray(plr, dtype=float)
        a, b = self.PLR_coeffs
        return np.clip(a + b * plr, 1.0, 2.0)

    def gross_sec_kwh_per_ton(self, plr, T_ambient_K):
        """Gross SEC [kWh/ton] with ambient and part-load corrections."""
        sec_amb = self.sec_ambient_corrected(T_ambient_K)
        plr_f = self.plr_factor(plr)
        return sec_amb * plr_f

    def cold_energy_recovery_kw(self, sendout_rate_ton_per_h, T_ambient_K):
        """Cold exergy recovery [kW] (Carnot estimate)."""
        m_ton_h = np.asarray(sendout_rate_ton_per_h, dtype=float)
        m_kg_s = m_ton_h * 1000.0 / 3600.0
        T_amb = np.asarray(T_ambient_K, dtype=float)
        carnot = 1.0 - self.T_storage / T_amb
        ex_cold = m_kg_s * self.h_fg * np.maximum(carnot, 0.0)
        return self.f_cold * ex_cold

    def net_power_kw(self, sendout_rate_ton_per_h, plr, T_ambient_K):
        """Net power consumption after cold recovery [kW]."""
        m = np.asarray(sendout_rate_ton_per_h, dtype=float)
        sec = self.gross_sec_kwh_per_ton(plr, T_ambient_K)
        P_gross = sec * m  # kW
        P_cold = self.cold_energy_recovery_kw(m, T_ambient_K)
        return np.maximum(P_gross - P_cold, 0.0)

    def net_sec_kwh_per_ton(self, plr, T_ambient_K, sendout_rate_ton_per_h=100.0):
        """Net SEC after cold recovery [kWh/ton]."""
        m = np.asarray(sendout_rate_ton_per_h, dtype=float)
        sec_gross = self.gross_sec_kwh_per_ton(plr, T_ambient_K)
        P_cold = self.cold_energy_recovery_kw(m, T_ambient_K)
        sec_cold = P_cold / np.where(m > 0, m, 1.0)
        return np.maximum(sec_gross - sec_cold, 0.0)

    def gas_sendout_kg_per_s(self, sendout_rate_ton_per_h):
        """Gas sendout mass flow [kg/s]."""
        return np.asarray(sendout_rate_ton_per_h, dtype=float) * 1000.0 / 3600.0

    def compute(self, sendout_rate_ton_per_h, plr, T_ambient_K=283.15):
        """Full computation returning all outputs."""
        m = sendout_rate_ton_per_h
        sec_gross = self.gross_sec_kwh_per_ton(plr, T_ambient_K)
        net_power = self.net_power_kw(m, plr, T_ambient_K)
        net_sec = self.net_sec_kwh_per_ton(plr, T_ambient_K, m)
        cold_kw = self.cold_energy_recovery_kw(m, T_ambient_K)
        gas_kg_s = self.gas_sendout_kg_per_s(m)

        return {
            "gross_sec_kwh_per_ton": sec_gross,
            "net_sec_kwh_per_ton": net_sec,
            "net_power_kw": net_power,
            "cold_recovery_kw": cold_kw,
            "gas_sendout_kg_per_s": gas_kg_s,
        }
