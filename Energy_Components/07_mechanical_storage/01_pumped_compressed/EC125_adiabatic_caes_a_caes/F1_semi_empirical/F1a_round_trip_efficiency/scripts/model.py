"""
EC125 — Adiabatic Compressed Air Energy Storage (A-CAES) — F1a Round-Trip Model

Key difference from diabatic CAES (EC123):
  - Heat of compression is stored in thermal energy storage (TES) during charging
  - Stored heat is returned to expand air during discharge — NO supplemental fuel
  - TES round-trip efficiency (eta_tes) governs how much heat is recovered
  - Round-trip efficiency: ~65-72% (vs 42-55% for diabatic EC123)

Cavern model (identical to EC123 — constant volume, isothermal):
    m(p)     = p * V / (R * T)
    m_max    = p_max * V / (R * T)
    m_min    = p_min * V / (R * T)
    SOC      = (m - m_min) / (m_max - m_min)

Charge (compression — heat stored in TES):
    P_elec_in = m_dot_air * w_comp / (eta_comp * eta_motor)    [kW]
    Q_stored  = P_elec_in * eta_comp * eta_motor                [kW_th]  (heat to TES)

Discharge (expansion — heat retrieved from TES):
    P_elec_out = m_dot_air * w_exp_ad * eta_exp * eta_gen       [kW]
    Q_fuel_in  = 0   (no fuel needed — heat from TES)

Round-trip efficiency (electricity only, no fuel):
    eta_RT = (E_out_elec) / E_in_elec
           = (w_exp_ad * eta_exp * eta_gen) / (w_comp / (eta_comp * eta_motor))

The TES efficiency is embedded in w_exp_adiabatic:
    w_exp_adiabatic = w_exp_ideal * eta_tes   (heat returned fraction determines expansion work)

References:
    Barbour, E., et al. (2015). A review on pumped power storage systems.
    Renewable and Sustainable Energy Reviews, 45, 598-614.
    Budt, M., Wolf, D., Span, R., Yan, J. (2016). A review on compressed air
    energy storage. Applied Energy, 170, 250-268.
    Wolf, D. & Budt, M. (2014). LTA-CAES: A low-temperature approach to adiabatic CAES.
    Applied Energy, 125, 158-164.
"""

import numpy as np


class ACAESF1a:
    """Adiabatic CAES — round-trip semi-empirical model (no supplemental fuel)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta_comp  = u["eta_compressor"]["value"]
        self.eta_motor = u["eta_motor"]["value"]
        self.eta_exp   = u["eta_expander"]["value"]
        self.eta_gen   = u["eta_generator"]["value"]
        self.eta_tes   = u["eta_tes"]["value"]
        self.heat_rate = u["heat_rate"]["value"]          # kJ/kWh_e — always 0 for A-CAES
        self.p_max     = u["p_max"]["value"]              # Pa
        self.p_min     = u["p_min"]["value"]              # Pa
        self.V         = u["cavern_volume"]["value"]      # m3
        self.T         = u["T_cavern"]["value"]           # K
        self.w_comp    = u["specific_work"]["value"]      # kJ/kg
        self.w_exp     = u["specific_expansion_adiabatic"]["value"]  # kJ/kg
        self.R         = u["R_air"]["value"]              # J/(kg.K)
        # Derived cavern limits
        self.m_max    = self.p_max * self.V / (self.R * self.T)
        self.m_min    = self.p_min * self.V / (self.R * self.T)
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
        """Electrical output power [kW] for given expander air mass-flow [kg/s].
        TES-recovered heat drives expansion — no fuel input.
        """
        m_dot = np.asarray(m_dot_air, dtype=float)
        return m_dot * self.w_exp * self.eta_exp * self.eta_gen

    def fuel_power(self, p_elec_out):
        """Fuel thermal input [kW] — always zero for A-CAES."""
        return np.zeros_like(np.asarray(p_elec_out, dtype=float))

    def fuel_mass_flow(self, p_elec_out):
        """Fuel mass-flow [kg/s] — always zero for A-CAES."""
        return np.zeros_like(np.asarray(p_elec_out, dtype=float))

    # ---------- capacity & efficiency ----------

    def energy_capacity_kwh(self):
        """Usable electrical energy capacity [kWh]."""
        return self.m_usable * self.w_exp * self.eta_exp * self.eta_gen / 3600.0

    def round_trip_efficiency(self):
        """
        A-CAES round-trip efficiency (electricity only — no fuel):
            eta_RT = E_elec_out / E_elec_in
                   = (w_exp * eta_exp * eta_gen) / (w_comp / (eta_comp * eta_motor))
        Expected: ~0.65-0.72 (significantly higher than diabatic EC123 ~0.42-0.55).
        """
        E_out     = self.w_exp * self.eta_exp * self.eta_gen / 3600.0       # kWh/kg
        E_in_elec = self.w_comp / (self.eta_comp * self.eta_motor) / 3600.0  # kWh/kg
        return E_out / E_in_elec

    def electric_round_trip_efficiency(self):
        """Alias for round_trip_efficiency (no fuel, identical)."""
        return self.round_trip_efficiency()

    def tes_heat_stored_kw(self, m_dot_air):
        """
        Rate of heat stored in TES during charging [kW_th].
        Approximately equal to compression work * compressor efficiency.
        """
        m_dot = np.asarray(m_dot_air, dtype=float)
        return m_dot * self.w_comp * self.eta_comp  # kJ/kg * kg/s = kW

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
            E_elec_in  = P * dt                                        # kWh
            E_to_air_kj = E_elec_in * 3600.0 * (self.eta_comp * self.eta_motor)
            m_added = E_to_air_kj / self.w_comp
            m_new   = self.air_mass(s) + m_added
            return float(self.soc_from_mass(min(m_new, self.m_max)))
        elif mode == "discharge":
            E_elec_out    = P * dt                                     # kWh
            E_from_air_kj = E_elec_out * 3600.0 / (self.eta_exp * self.eta_gen)
            m_removed = E_from_air_kj / self.w_exp
            m_new     = self.air_mass(s) - m_removed
            return float(self.soc_from_mass(max(m_new, self.m_min)))
        else:
            raise ValueError(f"mode must be 'charge', 'discharge', or 'idle', got '{mode}'")
