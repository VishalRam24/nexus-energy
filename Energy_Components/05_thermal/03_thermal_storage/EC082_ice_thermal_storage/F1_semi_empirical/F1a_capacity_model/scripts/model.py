"""
EC082 — Ice Thermal Storage — F1a Capacity Model

Latent ice TES at the water/ice phase-change temperature (0 C).  At F1a we
do not resolve the temperature field; we track only the state of charge as
the fraction of stored mass that is frozen, and apply rate limits that
realistically depend on SOC:

    SOC = m_ice / m_total                  (0 = all liquid, 1 = all ice)

    Energy stored [kWh] = SOC * capacity_kwh
    Q_charge_max(SOC)    = Q_charge_rated * (1 - SOC)         (slows as ice grows)
    Q_discharge_max(SOC) = Q_discharge_rated * SOC             (limited by remaining ice)
    Q_loss              = UA * (T_amb - T_phase)               (heat in raises losses)

Energy balance (per unit time, in J/s):

    dE/dt = q_charge_eff - q_discharge_eff - Q_loss
    dSOC/dt = dE/dt / (m_water * h_fusion)

where Q_loss melts ice (parasitic discharge).  Round-trip efficiency is
applied as an absorption factor on charging.

References:
    ASHRAE Handbook — HVAC Systems and Equipment (2020), ch.51 Thermal Storage.
    Dincer & Rosen (2021), Thermal Energy Storage, 3rd ed., Wiley.
"""

import numpy as np


class IceTESF1a:
    """Latent ice thermal storage — capacity-tracking model at fixed T_phase."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.cap_kwh    = u["capacity_kwh"]["value"]
        self.T_phase    = u["T_phase_change"]["value"]
        self.h_fus      = u["h_fusion"]["value"] * 1000.0   # kJ/kg -> J/kg
        self.m_water    = u["mass_water_kg"]["value"]
        self.Q_chg_max  = u["Q_charge_max"]["value"]    * 1000.0   # kW -> W
        self.Q_dis_max  = u["Q_discharge_max"]["value"] * 1000.0
        self.UA         = u["UA_loss"]["value"]
        self.eta_rt     = u["round_trip_efficiency"]["value"]

    # ------------------------------------------------------------------
    def heat_loss(self, T_amb):
        """Heat ingress from ambient (positive when T_amb > T_phase) [W]."""
        T_amb = np.asarray(T_amb, dtype=float)
        return self.UA * (T_amb - self.T_phase)

    def max_charge(self, soc):
        soc = np.asarray(soc, dtype=float)
        return self.Q_chg_max * np.clip(1.0 - soc, 0.0, 1.0)

    def max_discharge(self, soc):
        soc = np.asarray(soc, dtype=float)
        return self.Q_dis_max * np.clip(soc, 0.0, 1.0)

    def energy_stored_kwh(self, soc):
        soc = np.asarray(soc, dtype=float)
        return np.clip(soc, 0.0, 1.0) * self.cap_kwh

    def dSOC_dt(self, soc, q_charge, q_discharge, T_amb):
        """Rate of state-of-charge change [1/s]."""
        soc = np.asarray(soc, dtype=float)
        q_c = np.asarray(q_charge,    dtype=float)   # W requested charge
        q_d = np.asarray(q_discharge, dtype=float)   # W requested discharge

        q_c_eff = np.minimum(q_c, self.max_charge(soc)) * self.eta_rt
        q_d_eff = np.minimum(q_d, self.max_discharge(soc))
        Q_loss = self.heat_loss(T_amb)

        net_W = q_c_eff - q_d_eff - Q_loss
        E_capacity_J = self.m_water * self.h_fus
        return net_W / E_capacity_J

    def operating_state(self, soc, q_charge, q_discharge, T_amb):
        """Return all derived rates and effective powers (used by predict)."""
        q_c_eff = np.minimum(np.asarray(q_charge, dtype=float),
                              self.max_charge(soc)) * self.eta_rt
        q_d_eff = np.minimum(np.asarray(q_discharge, dtype=float),
                              self.max_discharge(soc))
        Q_loss = self.heat_loss(T_amb)
        return {
            "q_charge_effective_w":    q_c_eff,
            "q_discharge_effective_w": q_d_eff,
            "heat_loss_w":             Q_loss,
            "max_charge_w":            self.max_charge(soc),
            "max_discharge_w":         self.max_discharge(soc),
        }
