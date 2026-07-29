"""
EC112 -- Micro Gas Turbine -- F1b Part-Load + Ambient Temperature + Altitude

Extends F1a by adding:
  1. ISO power correction (ambient T and pressure effect on air density)
  2. Efficiency sensitivity to ambient temperature (~0.01/K, much stronger than
     reciprocating engines -- small single-shaft turbines have less pressure ratio
     margin and no turbocharger to compensate)
  3. Altitude derating (reduced air density at elevation)

Part-load efficiency:
    eta_el(PLR) = eta_el_rated * f_PLR(PLR)
    f_PLR = b0 + b1*PLR + b2*PLR^2

Ambient temperature correction (ISO air-density):
    f_power_amb(T_amb, P_amb) = (P_amb / P_ref) * sqrt(T_ref / T_amb)

Efficiency temperature correction:
    f_eta_T(T_amb) = 1 - f_amb_coeff * (T_amb - T_ref)
    (Strong sensitivity: ~1%/K degradation; at 40 degC vs ISO 15 degC,
     efficiency drops by ~25%*1%/K = ~25% relative.)
    Note: f_amb_coeff = 0.01/K means 1%/K (absolute) sensitivity.

Altitude correction:
    f_alt = 1 - alt_derate * altitude_m / 100
    (1%/100m for non-turbocharged or minimally-turbocharged micro gas turbines)

Power output:
    P_el = P_el_rated * PLR * f_power_amb * f_alt

Efficiency:
    eta_el = eta_el_rated * f_PLR * f_eta_T

References:
    US EPA CHP Catalog (2017), Section 5: Microturbines.
    Capstone C200 product data sheet (2023).
    ISO 2314:2009 Gas turbines -- Acceptance tests.
    Liss, W.E. et al. (2001). Naturally occurring high-inert-content gases and
        their impact on compression and combustion equipment. GRI Report.
"""

import numpy as np


class MicroGasTurbineF1b:
    """Recuperated micro gas turbine -- part-load + ambient + altitude model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_el_rated   = u["P_el_rated_kw"]["value"]              # kW_e
        self.eta_el_rated = u["eta_el_rated"]["value"]
        self.PLR_min      = u["PLR_min"]["value"]
        self.LHV          = u["LHV_gas_mjkg"]["value"]               # MJ/kg
        self.rho_gas      = u["rho_gas_kgm3"]["value"]               # kg/m^3
        self.b0           = u["b0"]["value"]
        self.b1           = u["b1"]["value"]
        self.b2           = u["b2"]["value"]
        self.T_ref        = u["T_ref_k"]["value"]                    # K
        self.P_ref        = u["P_ref_kpa"]["value"]                  # kPa
        self.f_amb_coeff  = u["f_amb_coeff"]["value"]                # 1/K
        self.alt_derate   = u["altitude_derating_pct_per_100m"]["value"] / 100.0  # fraction/100m

    # ------------------------------------------------------------------
    # Correction factors
    # ------------------------------------------------------------------

    def f_power_ambient(self, T_amb_k, P_amb_kpa=101.325):
        """
        ISO power correction factor (air-density effect).
        f = (P_amb / P_ref) * sqrt(T_ref / T_amb)
        """
        T = np.asarray(T_amb_k, dtype=float)
        P = np.asarray(P_amb_kpa, dtype=float)
        return (P / self.P_ref) * np.sqrt(self.T_ref / T)

    def f_eta_temperature(self, T_amb_k):
        """
        Efficiency correction for ambient temperature.
        Micro gas turbines are strongly sensitive (~0.01/K, or 1%/K absolute).
        At ISO (15 degC = 288.15 K): f = 1.0
        At 40 degC (313.15 K): f = 1 - 0.01 * (313.15 - 288.15) = 1 - 0.25 = 0.75
        Clipped to prevent unphysically low efficiency.
        """
        T = np.asarray(T_amb_k, dtype=float)
        f = 1.0 - self.f_amb_coeff * (T - self.T_ref)
        return np.clip(f, 0.3, 1.2)

    def f_altitude(self, altitude_m):
        """
        Altitude power derating factor (reduced air mass flow).
        f_alt = 1 - alt_derate * altitude_m / 100
        """
        alt = np.asarray(altitude_m, dtype=float)
        delta = np.maximum(0.0, alt)
        return np.clip(1.0 - self.alt_derate * delta / 100.0, 0.3, 1.0)

    # ------------------------------------------------------------------
    # Efficiency
    # ------------------------------------------------------------------

    def f_plr(self, PLR):
        """Part-load efficiency correction factor."""
        p = np.asarray(PLR, dtype=float)
        return self.b0 + self.b1 * p + self.b2 * p ** 2

    def eta_electrical(self, PLR, T_amb_k=288.15):
        """
        Net electrical efficiency at given PLR and ambient temperature.
        eta = eta_rated * f_PLR(PLR) * f_eta_T(T_amb)
        """
        p = np.asarray(PLR, dtype=float)
        eta = self.eta_el_rated * self.f_plr(p) * self.f_eta_temperature(T_amb_k)
        return np.clip(eta, 1e-6, 0.40)

    # ------------------------------------------------------------------
    # Power / heat flows
    # ------------------------------------------------------------------

    def power_electrical_kw(self, PLR, T_amb_k=288.15, P_amb_kpa=101.325, altitude_m=0.0):
        """Electrical output [kW_e]."""
        p = np.asarray(PLR, dtype=float)
        f_amb = self.f_power_ambient(T_amb_k, P_amb_kpa)
        f_alt = self.f_altitude(altitude_m)
        return self.P_el_rated * p * f_amb * f_alt

    def fuel_input_kw(self, PLR, T_amb_k=288.15, P_amb_kpa=101.325, altitude_m=0.0):
        """Fuel input power [kW_fuel, LHV]."""
        P_el = self.power_electrical_kw(PLR, T_amb_k, P_amb_kpa, altitude_m)
        eta_el = self.eta_electrical(PLR, T_amb_k)
        return np.where(eta_el > 1e-6, P_el / eta_el, 0.0)

    def gas_mass_flow_kgs(self, PLR, T_amb_k=288.15, P_amb_kpa=101.325, altitude_m=0.0):
        """Natural gas mass flow rate [kg/s]."""
        fuel_kw = self.fuel_input_kw(PLR, T_amb_k, P_amb_kpa, altitude_m)
        return fuel_kw / (self.LHV * 1000.0)

    def gas_volume_flow_m3h(self, PLR, T_amb_k=288.15, P_amb_kpa=101.325, altitude_m=0.0):
        """Natural gas volume flow rate [m^3/h]."""
        m_dot = self.gas_mass_flow_kgs(PLR, T_amb_k, P_amb_kpa, altitude_m)
        return m_dot / self.rho_gas * 3600.0

    def heat_rate_kj_kwh(self, PLR, T_amb_k=288.15):
        """Heat rate [kJ/kWh]."""
        eta = self.eta_electrical(PLR, T_amb_k)
        return np.where(eta > 1e-6, 3600.0 / eta, 0.0)
