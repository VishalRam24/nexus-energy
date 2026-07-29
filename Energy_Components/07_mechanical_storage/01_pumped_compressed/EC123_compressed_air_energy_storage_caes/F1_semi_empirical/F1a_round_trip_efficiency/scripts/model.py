"""
EC123 — Compressed Air Energy Storage (CAES) — F1a Round-Trip Model

Cavern (constant volume, isothermal) — air mass from ideal gas law:
    m(p)   = p * V / (R * T)             [kg]
    m_max  = p_max * V / (R * T)
    m_min  = p_min * V / (R * T)
    SOC    = (m - m_min) / (m_max - m_min)

Energy capacity (usable, electrical, diabatic with reheat):
    E_cap_kwh = (m_max - m_min) * w_exp * eta_expander * eta_generator / 3600

Charge (compression):
    P_elec_in = m_dot_air * w_comp / (eta_compressor * eta_motor)   [kW]

Discharge (expansion + supplemental fuel for diabatic mode):
    P_elec_out = m_dot_air * w_exp * eta_expander * eta_generator    [kW]
    Q_fuel_in  = P_elec_out * heat_rate / 3600                       [kW thermal]
    m_dot_fuel = Q_fuel_in / LHV                                     [kg/s]

Round-trip efficiency:
    Diabatic (with fuel input — electricity-only ratio):
        eta_RT = (E_out_elec) / (E_in_elec + E_in_fuel)
    Electricity-only RTE:
        eta_RT_elec = (E_out_elec) / E_in_elec   (can exceed 1 since fuel adds energy)
    For F1a we report the all-energy (electricity + fuel) round-trip efficiency.

References:
    Budt, M., Wolf, D., Span, R., Yan, J. (2016). A review on compressed air
    energy storage. Applied Energy, 170, 250-268.
    Luo, X. et al. (2015). Overview of current development in electrical energy
    storage technologies. Applied Energy, 137, 511-536.
"""

import numpy as np


class CAESF1a:
    """Compressed Air Energy Storage — round-trip semi-empirical model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta_comp = u["eta_compressor"]["value"]
        self.eta_motor = u["eta_motor"]["value"]
        self.eta_exp = u["eta_expander"]["value"]
        self.eta_gen = u["eta_generator"]["value"]
        self.heat_rate = u["heat_rate"]["value"]      # kJ/kWh_e
        self.fuel_lhv = u["fuel_lhv"]["value"]        # kJ/kg
        self.p_max = u["p_max"]["value"]              # Pa
        self.p_min = u["p_min"]["value"]              # Pa
        self.V = u["cavern_volume"]["value"]          # m3
        self.T = u["T_cavern"]["value"]               # K
        self.w_comp = u["specific_work"]["value"]     # kJ/kg
        self.w_exp = u["specific_expansion"]["value"] # kJ/kg
        self.R = u["R_air"]["value"]                  # J/(kg.K)
        # Derived
        self.m_max = self.p_max * self.V / (self.R * self.T)
        self.m_min = self.p_min * self.V / (self.R * self.T)
        self.m_usable = self.m_max - self.m_min

    # ---------- cavern state ----------
    def air_mass(self, soc):
        """Air mass [kg] in cavern at given SOC."""
        s = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        return self.m_min + s * self.m_usable

    def cavern_pressure(self, soc):
        """Cavern pressure [Pa] at given SOC."""
        m = self.air_mass(soc)
        return m * self.R * self.T / self.V

    def soc_from_mass(self, mass):
        """SOC from air mass."""
        m = np.asarray(mass, dtype=float)
        return np.clip((m - self.m_min) / self.m_usable, 0.0, 1.0)

    # ---------- power flows ----------
    def charge_power(self, m_dot_air):
        """Electrical input power [kW] for given air mass-flow [kg/s] during charging."""
        m_dot = np.asarray(m_dot_air, dtype=float)
        return m_dot * self.w_comp / (self.eta_comp * self.eta_motor)

    def discharge_power(self, m_dot_air):
        """Electrical output power [kW] for given expander air mass-flow [kg/s]."""
        m_dot = np.asarray(m_dot_air, dtype=float)
        return m_dot * self.w_exp * self.eta_exp * self.eta_gen

    def fuel_power(self, p_elec_out):
        """Fuel thermal input rate [kW] for diabatic discharge."""
        p = np.asarray(p_elec_out, dtype=float)
        return p * self.heat_rate / 3600.0

    def fuel_mass_flow(self, p_elec_out):
        """Fuel mass-flow rate [kg/s] for diabatic discharge."""
        return self.fuel_power(p_elec_out) / self.fuel_lhv

    # ---------- capacity & efficiency ----------
    def energy_capacity_kwh(self):
        """Usable electrical energy capacity [kWh]."""
        return self.m_usable * self.w_exp * self.eta_exp * self.eta_gen / 3600.0

    def round_trip_efficiency(self):
        """
        Diabatic round-trip efficiency including fuel input:
            eta = E_out_elec / (E_in_elec + E_in_fuel)
        """
        E_out = self.w_exp * self.eta_exp * self.eta_gen / 3600.0           # kWh per kg air
        E_in_elec = self.w_comp / (self.eta_comp * self.eta_motor) / 3600.0  # kWh/kg
        E_in_fuel = E_out * self.heat_rate / 3600.0                          # kWh/kg
        return E_out / (E_in_elec + E_in_fuel)

    def electric_round_trip_efficiency(self):
        """
        Electricity-only round-trip ratio E_out_elec / E_in_elec.
        This can exceed 1 for diabatic CAES because fuel adds energy.
        """
        E_out = self.w_exp * self.eta_exp * self.eta_gen
        E_in_elec = self.w_comp / (self.eta_comp * self.eta_motor)
        return E_out / E_in_elec

    # ---------- SOC update from power command ----------
    def soc_update(self, soc0, power_kw, dt_hours, mode):
        """
        Update SOC for a single time step under power command.
        mode: 'charge' (P_elec_in), 'discharge' (P_elec_out), or 'idle'.
        Returns new SOC clamped to [0, 1].
        """
        s = float(np.clip(soc0, 0.0, 1.0))
        P = float(power_kw)
        dt = float(dt_hours)
        if mode == "idle" or P <= 0.0 or dt <= 0.0:
            return s
        if mode == "charge":
            # Electric energy in -> air added
            E_elec_in = P * dt                             # kWh
            E_to_air_kj = E_elec_in * 3600.0 * (self.eta_comp * self.eta_motor)
            m_added = E_to_air_kj / self.w_comp            # kg
            m_new = self.air_mass(s) + m_added
            return float(self.soc_from_mass(min(m_new, self.m_max)))
        elif mode == "discharge":
            E_elec_out = P * dt                            # kWh
            E_from_air_kj = E_elec_out * 3600.0 / (self.eta_exp * self.eta_gen)
            m_removed = E_from_air_kj / self.w_exp         # kg
            m_new = self.air_mass(s) - m_removed
            return float(self.soc_from_mass(max(m_new, self.m_min)))
        else:
            raise ValueError(f"mode must be 'charge', 'discharge', or 'idle', got '{mode}'")
