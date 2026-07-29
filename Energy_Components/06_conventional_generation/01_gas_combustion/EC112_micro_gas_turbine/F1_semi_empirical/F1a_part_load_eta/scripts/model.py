"""
EC112 — Micro Gas Turbine — F1a Part-Load Efficiency Model

Recuperated micro gas turbine (30-300 kWe range, e.g. Capstone C200).

  eta_el(PLR, T_amb) = eta_el_rated * f_PLR(PLR) * f_amb(T_amb)
  f_PLR              = b0 + b1*PLR + b2*PLR^2
  f_amb              = 1 - f_amb_coeff*(T_amb - T_ref)
  P_el               = P_el_rated * PLR * f_amb                 [kW_e]
  fuel_kw            = P_el / eta_el                            [kW_fuel]
  m_dot_gas          = fuel_kw / (LHV * 1000)                   [kg/s]
  v_dot_gas          = m_dot_gas / rho_gas * 3600               [m^3/h]
  heat_rate          = 3600 / eta_el                            [kJ/kWh]

References:
    US EPA CHP Catalog (2017), Section 5: Microturbines.
    Capstone C200 product data sheet.
"""

import numpy as np


class MicroGasTurbineF1a:
    """Recuperated micro gas turbine — part-load efficiency model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_el_rated = u["P_el_rated"]["value"]              # kW_e
        self.eta_el_rated = u["eta_el_rated"]["value"]
        self.PLR_min = u["PLR_min"]["value"]
        self.LHV = u["LHV_gas_mjkg"]["value"]                   # MJ/kg
        self.rho_gas = u["rho_gas_kgm3"]["value"]               # kg/m^3
        self.b0 = u["b0"]["value"]
        self.b1 = u["b1"]["value"]
        self.b2 = u["b2"]["value"]
        self.T_ref = u["T_ref_c"]["value"]
        self.f_amb_coeff = u["f_amb_coeff"]["value"]

    def _f_plr(self, plr):
        p = np.asarray(plr, dtype=float)
        return self.b0 + self.b1 * p + self.b2 * p**2

    def _f_amb(self, T_amb):
        T = np.asarray(T_amb, dtype=float)
        return np.clip(1.0 - self.f_amb_coeff * (T - self.T_ref), 0.5, 1.2)

    def eta_electrical(self, plr, T_amb=15.0):
        """Net electrical efficiency at given PLR and ambient T."""
        eta = self.eta_el_rated * self._f_plr(plr) * self._f_amb(T_amb)
        return np.clip(eta, 1e-6, 0.40)

    def compute(self, part_load_ratio, ambient_temp_c=15.0):
        """Compute power, fuel and efficiency at given PLR and ambient T.

        Returns
        -------
        dict: electrical_power_kw, fuel_input_kw, eta_electrical,
              gas_mass_flow_kgs, gas_volume_flow_m3h, heat_rate_kjkwh
        """
        plr = np.clip(np.asarray(part_load_ratio, dtype=float), self.PLR_min, 1.0)
        f_amb = self._f_amb(ambient_temp_c)

        eta_el = self.eta_electrical(plr, ambient_temp_c)
        P_el = self.P_el_rated * plr * f_amb                              # kW_e
        fuel_kw = np.where(eta_el > 1e-6, P_el / eta_el, 0.0)             # kW_fuel
        m_dot = fuel_kw / (self.LHV * 1000.0)                             # kg/s
        v_dot = m_dot / self.rho_gas * 3600.0                             # m^3/h
        hr = np.where(eta_el > 1e-6, 3600.0 / eta_el, 0.0)                # kJ/kWh

        return {
            "electrical_power_kw": P_el,
            "fuel_input_kw": fuel_kw,
            "eta_electrical": eta_el,
            "gas_mass_flow_kgs": m_dot,
            "gas_volume_flow_m3h": v_dot,
            "heat_rate_kjkwh": hr,
        }
