"""
EC190 — LNG Regasification Terminal — F1a Energy Model

The model computes electricity/heat demand and gas sendout for LNG regasification.

Energy balance per tonne of LNG:
    Q_regas = h_fg + c_p_gas * (T_sendout - T_storage)   [kJ/kg]
               ≈ 510 kJ/kg (latent) + sensible heat

Specific Energy Consumption (SEC):
    SEC [kWh/ton] depends on heat source:
        - Open Rack Vaporiser (seawater): 20–35 kWh/ton
        - Submerged Combustion Vaporiser (SCV): 60–100 kWh/ton
        - Ambient Air Vaporiser (AAV): 25–45 kWh/ton
    Default: 50 kWh/ton (intermediate / mixed system)

Power demand:
    P_elec [kW] = SEC [kWh/ton] × sendout_rate [ton/h]

Gas sendout:
    Q_gas [kg/s] = sendout_rate [ton/h] × 1000 / 3600

Cold energy recovery (optional):
    Available cold exergy at LNG vaporisation:
        Ex_cold [kW] = m_dot * T_env * (s_gas - s_LNG) - (h_gas - h_LNG) / T_env
    Simplified: Ex_cold ≈ m_dot * Δh * (1 - T_storage/T_env) / 3600
    Power recovered: P_cold [kW] = f_cold * Ex_cold

Net power consumption:
    P_net [kW] = P_elec - P_cold

References:
    Mokhatab, S. et al. (2014). Handbook of Liquefied Natural Gas. Elsevier.
    Shah, N. et al. (2013). J. Natural Gas Science & Engineering, 11, 9-16.
"""

import numpy as np


class LNGRegasF1a:
    """SEC-based LNG regasification energy model with optional cold recovery."""

    def __init__(self, params: dict):
        t = params["terminal"]
        g = params["gas"]

        self.sec_base = t["sec_base"]["value"]             # kWh/ton
        self.f_cold = t["cold_energy_recovery_fraction"]["value"]
        self.rho_LNG = t["LNG_density"]["value"]           # kg/m³
        self.h_fg = t["vaporisation_heat"]["value"]        # kJ/kg
        self.LHV = g["LHV"]["value"]                       # MJ/kg
        self.T_storage = g["T_storage"]["value"]           # K (-162°C)
        self.T_sendout = g["T_sendout"]["value"]           # K (5°C)

    def power_demand_kw(self, sendout_rate_ton_per_h, sec_kwh_per_ton=None):
        """Total electrical/heat power demand [kW]."""
        m = np.asarray(sendout_rate_ton_per_h, dtype=float)
        sec = self.sec_base if sec_kwh_per_ton is None else np.asarray(sec_kwh_per_ton, dtype=float)
        return sec * m  # kW (SEC [kWh/ton] × flow [ton/h] = kW)

    def cold_energy_recovery_kw(self, sendout_rate_ton_per_h, T_ambient_K=288.15,
                                f_cold=None):
        """
        Available cold exergy recovery power [kW].

        Simplified Carnot-based estimate:
            P_cold = f_cold * m_dot [kg/s] * h_fg [kJ/kg] * (1 - T_storage/T_amb)
        """
        m_ton_h = np.asarray(sendout_rate_ton_per_h, dtype=float)
        m_kg_s = m_ton_h * 1000.0 / 3600.0
        T_amb = np.asarray(T_ambient_K, dtype=float)
        fc = self.f_cold if f_cold is None else np.asarray(f_cold, dtype=float)
        carnot_factor = 1.0 - self.T_storage / T_amb
        ex_cold_kw = m_kg_s * self.h_fg * carnot_factor  # kW
        return fc * np.maximum(ex_cold_kw, 0.0)

    def net_power_kw(self, sendout_rate_ton_per_h, sec_kwh_per_ton=None,
                     T_ambient_K=288.15, f_cold=None):
        """Net power consumption after cold energy recovery [kW]."""
        P_gross = self.power_demand_kw(sendout_rate_ton_per_h, sec_kwh_per_ton)
        P_cold = self.cold_energy_recovery_kw(sendout_rate_ton_per_h, T_ambient_K, f_cold)
        return P_gross - P_cold

    def gas_sendout_kg_per_s(self, sendout_rate_ton_per_h):
        """Gas sendout as mass flow [kg/s]."""
        m = np.asarray(sendout_rate_ton_per_h, dtype=float)
        return m * 1000.0 / 3600.0

    def gas_sendout_m3_per_day(self, sendout_rate_ton_per_h):
        """Gas sendout in standard m³/day (density at STP ≈ 0.717 kg/m³ for NG)."""
        rho_std = 0.717  # kg/m³ at standard conditions (typical NG)
        m_kg_s = self.gas_sendout_kg_per_s(sendout_rate_ton_per_h)
        return m_kg_s / rho_std * 86400.0

    def net_sec_kwh_per_ton(self, sendout_rate_ton_per_h, sec_kwh_per_ton=None,
                             T_ambient_K=288.15, f_cold=None):
        """Net SEC after cold energy recovery [kWh/ton]."""
        sec = self.sec_base if sec_kwh_per_ton is None else np.asarray(sec_kwh_per_ton, dtype=float)
        m = np.asarray(sendout_rate_ton_per_h, dtype=float)
        P_cold = self.cold_energy_recovery_kw(m, T_ambient_K, f_cold)
        P_cold_sec = P_cold / m  # kW / (ton/h) = kWh/ton
        return np.maximum(sec - P_cold_sec, 0.0)
