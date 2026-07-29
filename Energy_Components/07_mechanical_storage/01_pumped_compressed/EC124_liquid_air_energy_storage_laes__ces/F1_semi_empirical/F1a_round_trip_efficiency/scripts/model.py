"""
EC124 — Liquid Air Energy Storage (LAES / CES) — F1a Round-Trip Model

Charge (liquefaction):
    P_in   = m_dot_liq * w_liq / eta_liquefier        [kW]
    where w_liq is specific electrical work for liquefaction [kWh/kg],
    converted: P_in [kW] = m_dot [kg/s] * w_liq [kWh/kg] * 3600 / eta_liquefier

Discharge (pump → evaporate → expand → generate):
    P_out  = m_dot_liq * w_disch * eta_pump * eta_expander * eta_generator * 3600   [kW]

Storage SOC (cryogenic tank):
    SOC = m_liquid / m_tank_max

Boil-off (self-discharge during idle):
    dm/dt = -k_bo * m_liquid     =>     m(t) = m0 * exp(-k_bo * t)
    k_bo = boil_off_per_day / 24 (per hour)

Energy capacity:
    E_cap_kwh = m_tank_max * w_disch * eta_pump * eta_expander * eta_generator

Round-trip efficiency:
    eta_RT = (w_disch * eta_pump * eta_expander * eta_generator) /
             (w_liq / eta_liquefier)
    Typical: 0.50-0.70 for stand-alone LAES.

References:
    Morgan, R., Nelmes, S., Gibson, E., Brett, G. (2015). Liquid air energy
    storage – Analysis and first results from a pilot scale demonstration plant.
    Applied Energy, 137, 845-853.
    Sciacovelli, A., Vecchi, A., Ding, Y. (2017). Liquid air energy storage (LAES)
    with packed bed cold thermal storage – From component to system level
    performance through dynamic modelling. Applied Energy, 190, 84-98.
"""

import numpy as np


class LAESF1a:
    """Liquid Air Energy Storage — round-trip semi-empirical model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.w_liq = u["specific_liquefaction_kwh_per_kg"]["value"]    # kWh/kg
        self.w_disch = u["specific_discharge_kwh_per_kg"]["value"]      # kWh/kg
        self.eta_liq = u["eta_liquefier"]["value"]
        self.eta_pump = u["eta_pump"]["value"]
        self.eta_exp = u["eta_expander"]["value"]
        self.eta_gen = u["eta_generator"]["value"]
        self.m_tank_max = u["tank_capacity_kg"]["value"]
        self.T_storage = u["T_storage"]["value"]
        self.boil_off_per_day = u["boil_off_per_day"]["value"]
        self.rho_liquid = u["rho_liquid_air"]["value"]
        self.k_bo = self.boil_off_per_day / 24.0  # per hour

    # ---------- state ----------
    def liquid_mass(self, soc):
        """Liquid air mass [kg] at given SOC."""
        s = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        return s * self.m_tank_max

    def soc_from_mass(self, mass):
        m = np.asarray(mass, dtype=float)
        return np.clip(m / self.m_tank_max, 0.0, 1.0)

    def tank_volume(self, soc):
        """Liquid volume in tank [m3]."""
        return self.liquid_mass(soc) / self.rho_liquid

    # ---------- power ----------
    def charge_power(self, m_dot_liquid):
        """Electrical input power [kW] to liquefy m_dot_liquid [kg/s]."""
        m_dot = np.asarray(m_dot_liquid, dtype=float)
        return m_dot * self.w_liq * 3600.0 / self.eta_liq

    def discharge_power(self, m_dot_liquid):
        """Electrical output power [kW] from expanding m_dot_liquid [kg/s]."""
        m_dot = np.asarray(m_dot_liquid, dtype=float)
        return m_dot * self.w_disch * 3600.0 * self.eta_pump * self.eta_exp * self.eta_gen

    # ---------- capacity / efficiency ----------
    def energy_capacity_kwh(self):
        """Usable electrical energy capacity [kWh] (full tank)."""
        return self.m_tank_max * self.w_disch * self.eta_pump * self.eta_exp * self.eta_gen

    def round_trip_efficiency(self):
        """Round-trip electrical efficiency [-]."""
        E_out = self.w_disch * self.eta_pump * self.eta_exp * self.eta_gen
        E_in = self.w_liq / self.eta_liq
        return E_out / E_in

    # ---------- self-discharge ----------
    def boil_off_loss(self, soc, time_hours):
        """Mass remaining [kg] after boil-off over time_hours."""
        m0 = self.liquid_mass(soc)
        t = np.asarray(time_hours, dtype=float)
        return m0 * np.exp(-self.k_bo * t)

    def soc_after_standby(self, soc, time_hours):
        return self.soc_from_mass(self.boil_off_loss(soc, time_hours))

    # ---------- SOC update ----------
    def soc_update(self, soc0, power_kw, dt_hours, mode):
        """Update SOC under power command. Returns new SOC clamped to [0,1]."""
        s = float(np.clip(soc0, 0.0, 1.0))
        P = float(power_kw)
        dt = float(dt_hours)
        if mode == "idle":
            return float(self.soc_after_standby(s, dt))
        if P <= 0.0 or dt <= 0.0:
            return s
        if mode == "charge":
            E_elec_in = P * dt                          # kWh
            m_added = E_elec_in * self.eta_liq / self.w_liq
            m_new = self.liquid_mass(s) + m_added
            return float(self.soc_from_mass(min(m_new, self.m_tank_max)))
        elif mode == "discharge":
            E_elec_out = P * dt                          # kWh
            denom = self.w_disch * self.eta_pump * self.eta_exp * self.eta_gen
            m_removed = E_elec_out / denom
            m_new = self.liquid_mass(s) - m_removed
            return float(self.soc_from_mass(max(m_new, 0.0)))
        else:
            raise ValueError(f"mode must be 'charge', 'discharge', or 'idle', got '{mode}'")
