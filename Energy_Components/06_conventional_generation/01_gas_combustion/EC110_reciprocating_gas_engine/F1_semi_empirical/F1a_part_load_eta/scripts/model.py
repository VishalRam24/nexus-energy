"""
EC110 — Reciprocating Gas Engine — F1a Part-Load Efficiency Model

Lean-burn natural gas reciprocating engine genset.

  eta_el(PLR) = eta_el_rated * (b0 + b1*PLR + b2*PLR^2)
  P_el        = P_el_rated * PLR * f_amb(T_amb)         [kW_e]
  fuel_kw     = P_el / eta_el                           [kW_fuel, LHV]
  m_dot_gas   = fuel_kw / (LHV * 1000)                  [kg/s]
  v_dot_gas   = m_dot_gas / rho_gas * 3600              [m^3/h]
  SFC         = m_dot_gas * 3600 * 1000 / P_el          [g/kWh]

References:
    US EPA CHP Catalog (2017), Section 2: Reciprocating Engines.
    Jenbacher J320/J420 product data sheets, INNIO.
"""

import numpy as np


class ReciprocatingGasEngineF1a:
    """Reciprocating gas engine — part-load efficiency model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_el_rated = u["P_el_rated"]["value"]            # kW_e
        self.eta_el_rated = u["eta_el_rated"]["value"]
        self.PLR_min = u["PLR_min"]["value"]
        self.LHV = u["LHV_gas_mjkg"]["value"]                 # MJ/kg
        self.rho_gas = u["rho_gas_kgm3"]["value"]             # kg/m^3
        self.b0 = u["b0"]["value"]
        self.b1 = u["b1"]["value"]
        self.b2 = u["b2"]["value"]
        self.T_ref = u["T_ref_c"]["value"]
        self.derating = u["derating_per_degC"]["value"]

    def _f_amb(self, T_amb):
        """Ambient derating factor (only above T_ref)."""
        T = np.asarray(T_amb, dtype=float)
        return np.clip(1.0 - self.derating * np.maximum(0.0, T - self.T_ref), 0.7, 1.0)

    def eta_electrical(self, plr):
        """Electrical efficiency as a function of PLR."""
        p = np.asarray(plr, dtype=float)
        eta = self.eta_el_rated * (self.b0 + self.b1 * p + self.b2 * p**2)
        return np.clip(eta, 0.0, 0.50)

    def compute(self, part_load_ratio, ambient_temp_c=25.0):
        """Compute power, fuel and efficiency at given PLR and ambient T.

        Returns
        -------
        dict: electrical_power_kw, fuel_input_kw, eta_electrical,
              gas_mass_flow_kgs, gas_volume_flow_m3h, sfc_gkwh
        """
        plr = np.clip(np.asarray(part_load_ratio, dtype=float), self.PLR_min, 1.0)
        f_amb = self._f_amb(ambient_temp_c)

        eta_el = self.eta_electrical(plr)
        P_el = self.P_el_rated * plr * f_amb                              # kW_e
        fuel_kw = np.where(eta_el > 1e-6, P_el / eta_el, 0.0)             # kW_fuel
        # mass flow: fuel_kw / (LHV [MJ/kg] * 1000 kJ/MJ) = kg/s
        m_dot = fuel_kw / (self.LHV * 1000.0)                             # kg/s
        v_dot = m_dot / self.rho_gas * 3600.0                             # m^3/h
        sfc = np.where(P_el > 1e-6, m_dot * 3600.0 * 1000.0 / P_el, 0.0)  # g/kWh

        return {
            "electrical_power_kw": P_el,
            "fuel_input_kw": fuel_kw,
            "eta_electrical": eta_el,
            "gas_mass_flow_kgs": m_dot,
            "gas_volume_flow_m3h": v_dot,
            "sfc_gkwh": sfc,
        }
