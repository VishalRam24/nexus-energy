"""
EC110 -- Reciprocating Gas Engine -- F1b Part-Load + Altitude + Ambient Temperature

Extends F1a by adding:
  1. Altitude derating (reduced air density at elevation)
  2. Ambient temperature derating (separate from F1a single combined factor)
  3. Efficiency degradation model at part load (quadratic)
  4. Specific fuel consumption output

Part-load electrical efficiency:
    eta_el(PLR) = eta_el_rated * (b0 + b1*PLR + b2*PLR^2)

Ambient temperature derating (above T_ref = 25 degC):
    f_temp = 1 - temp_derate * max(0, T_amb - T_ref)

Altitude derating (above sea level):
    f_alt = 1 - alt_derate * altitude_m / 100
    (Turbocharged engines: ~0.9%/100m; ISO 3046-1)

Combined derating:
    f_total = f_temp * f_alt

Power output:
    P_el = P_el_rated * PLR * f_total

Fuel input:
    fuel_kw = P_el / eta_el

Specific fuel consumption (LHV basis):
    SFC = 3600 / (eta_el * LHV_gas [kJ/kWh_fuel]) -- or from mass flow

References:
    US EPA CHP Catalog (2017), Section 2: Reciprocating Engines.
    ISO 3046-1:2002 Reciprocating internal combustion engines -- Part 1: Declarations of power.
    Jenbacher J320/J420 product data sheets, INNIO.
    Caterpillar G3500 Series gas engine specification data.
"""

import numpy as np


class ReciprocatingGasEngineF1b:
    """Reciprocating gas engine with part-load efficiency, altitude and ambient correction."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_el_rated    = u["P_el_rated_kw"]["value"]              # kW_e
        self.eta_el_rated  = u["eta_el_rated"]["value"]
        self.PLR_min       = u["PLR_min"]["value"]
        self.LHV           = u["LHV_gas_mjkg"]["value"]               # MJ/kg
        self.rho_gas       = u["rho_gas_kgm3"]["value"]               # kg/m^3
        self.b0            = u["b0"]["value"]
        self.b1            = u["b1"]["value"]
        self.b2            = u["b2"]["value"]
        self.T_ref         = u["T_ref_c"]["value"]
        self.temp_derate   = u["temp_derating_pct_per_degC"]["value"] / 100.0
        self.temp_start    = u["temp_derating_start_c"]["value"]
        self.alt_ref       = u["altitude_ref_m"]["value"]
        self.alt_derate    = u["altitude_derating_pct_per_100m"]["value"] / 100.0

    # ------------------------------------------------------------------
    # Derating factors
    # ------------------------------------------------------------------

    def f_temperature(self, T_amb_c):
        """
        Ambient temperature derating factor.
        Active above T_ref (25 degC); no boost below.
        """
        T = np.asarray(T_amb_c, dtype=float)
        excess = np.maximum(0.0, T - self.temp_start)
        return np.clip(1.0 - self.temp_derate * excess, 0.5, 1.0)

    def f_altitude(self, altitude_m):
        """
        Altitude derating factor (reduced air density / lower manifold pressure).
        f_alt = 1 - (alt_derate_pct/100m) * (altitude - alt_ref) / 100
        ISO 3046-1: ~0.9%/100m for turbocharged engines.
        """
        alt = np.asarray(altitude_m, dtype=float)
        delta = np.maximum(0.0, alt - self.alt_ref)
        return np.clip(1.0 - self.alt_derate * delta / 100.0, 0.5, 1.0)

    def f_combined(self, T_amb_c, altitude_m):
        """Combined temperature and altitude derating."""
        return self.f_temperature(T_amb_c) * self.f_altitude(altitude_m)

    # ------------------------------------------------------------------
    # Efficiency
    # ------------------------------------------------------------------

    def eta_electrical(self, PLR):
        """
        Electrical efficiency at given PLR (independent of ambient conditions).
        Efficiency curve peaks near full load for lean-burn gas engines.
        """
        p = np.asarray(PLR, dtype=float)
        f_plr = self.b0 + self.b1 * p + self.b2 * p ** 2
        eta = self.eta_el_rated * f_plr
        return np.clip(eta, 0.0, 0.50)

    # ------------------------------------------------------------------
    # Power / heat flows
    # ------------------------------------------------------------------

    def power_electrical_kw(self, PLR, T_amb_c=25.0, altitude_m=0.0):
        """Electrical output [kW_e]."""
        p = np.asarray(PLR, dtype=float)
        f = self.f_combined(T_amb_c, altitude_m)
        return self.P_el_rated * p * f

    def fuel_input_kw(self, PLR, T_amb_c=25.0, altitude_m=0.0):
        """Fuel input power [kW_fuel, LHV]."""
        P_el = self.power_electrical_kw(PLR, T_amb_c, altitude_m)
        eta_el = self.eta_electrical(PLR)
        return np.where(eta_el > 1e-6, P_el / eta_el, 0.0)

    def gas_mass_flow_kgs(self, PLR, T_amb_c=25.0, altitude_m=0.0):
        """Natural gas mass flow rate [kg/s]."""
        fuel_kw = self.fuel_input_kw(PLR, T_amb_c, altitude_m)
        # fuel_kw / (LHV [MJ/kg] * 1e3 kJ/MJ) = kg/s
        return fuel_kw / (self.LHV * 1000.0)

    def gas_volume_flow_m3h(self, PLR, T_amb_c=25.0, altitude_m=0.0):
        """Natural gas volume flow rate [m^3/h]."""
        m_dot = self.gas_mass_flow_kgs(PLR, T_amb_c, altitude_m)
        return m_dot / self.rho_gas * 3600.0

    def sfc_g_kwh(self, PLR, T_amb_c=25.0, altitude_m=0.0):
        """Specific fuel consumption [g/kWh] (LHV basis)."""
        P_el = self.power_electrical_kw(PLR, T_amb_c, altitude_m)
        m_dot = self.gas_mass_flow_kgs(PLR, T_amb_c, altitude_m)
        return np.where(P_el > 1e-6, m_dot * 3600.0 * 1000.0 / P_el, 0.0)

    def heat_rate_kj_kwh(self, PLR):
        """Electrical heat rate [kJ/kWh]."""
        eta = self.eta_electrical(PLR)
        return np.where(eta > 1e-6, 3600.0 / eta, 0.0)
