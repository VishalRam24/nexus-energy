"""
EC114 -- Supercritical / Ultra-Supercritical Coal Plant -- F1b Part-Load + Flue Loss

Extends F1a by adding:
  1. Stack temperature model (rises at part load, but USC air preheater
     controls it more tightly than subcritical)
  2. Explicit flue gas enthalpy loss term
  3. Auxiliary power part-load curve (USC fans/pumps degrade at turndown)

Technology comparison:
    Subcritical (EC113): eta_net ~35-38%, T_stack ~130 degC rated
    Supercritical (SC):  eta_net ~42-44%, T_stack ~125 degC rated
    Ultra-Supercritical: eta_net ~44-47%, T_stack ~120 degC rated
    -> USC uses lower excess air (15%) and advanced air preheater -> lower stack loss

CO2 intensity note:
    CO2 intensity is validated at rated load (PLR=1) only.
    RATIONALE: At part load the efficiency penalty causes CO2/kWh to spike
    naturally above the rated benchmark. The IEA/EPA reference condition
    (PLR=1, T_amb=15C) is the relevant engineering benchmark. See test suite
    RATIONALE comments. Typical USC at rated: 750-850 g/kWh.

References:
    Weitzel, P.S. (2011). Steam generator for advanced ultra supercritical
    power plants. ASME PVP-2011-57934.
    IEA Clean Coal Centre, CCC/168 (2010).
    Luo, Z., Zhu, S., Wang, R., Xu, G. (2013). Energy, 57, 236-244.
"""

import numpy as np


class SupercriticalCoalF1b:
    """SC/USC coal plant with part-load efficiency, flue loss, and auxiliary power."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated         = u["rated_power_mw"]["value"]
        self.eta_iso         = u["eta_iso"]["value"]
        self.T_amb_ref       = u["T_amb_ref"]["value"]
        self.k_amb           = u["k_amb"]["value"]
        self.a0              = u["plr_coeffs"]["a0"]["value"]
        self.a1              = u["plr_coeffs"]["a1"]["value"]
        self.a2              = u["plr_coeffs"]["a2"]["value"]
        self.LHV_coal        = u["LHV_coal"]["value"]
        self.CO2_per_kg_coal = u["CO2_per_kg_coal"]["value"]
        self.T_stack_rated   = u["T_stack_rated_c"]["value"]
        self.T_stack_offset  = u["T_stack_partload_offset_c"]["value"]
        self.cp_flue         = u["cp_flue_kj_kgK"]["value"]
        self.stoich_air      = u["stoich_air_kg_per_kg_coal"]["value"]
        self.excess_air      = u["excess_air_fraction"]["value"]
        self.aux_rated_frac  = u["aux_power_rated_pct"]["value"] / 100.0
        self.aux_extra_frac  = u["aux_power_partload_extra_pct"]["value"] / 100.0
        self.min_plr         = u["min_plr"]["value"]

    def f_plr(self, plr):
        plr = np.asarray(plr, dtype=float)
        return self.a0 + self.a1 * plr + self.a2 * plr ** 2

    def f_amb(self, T_amb):
        T_amb = np.asarray(T_amb, dtype=float)
        return 1.0 - self.k_amb * (T_amb - self.T_amb_ref)

    def efficiency_net(self, plr, T_amb):
        return self.eta_iso * self.f_plr(plr) * self.f_amb(T_amb)

    def stack_temperature_c(self, plr):
        plr = np.asarray(plr, dtype=float)
        return self.T_stack_rated + self.T_stack_offset * (1.0 - plr)

    def coal_rate_kgs(self, plr, T_amb):
        P_net    = self.power_mw(plr)
        eta_net  = self.efficiency_net(plr, T_amb)
        eta_safe = np.where(np.asarray(eta_net) > 1e-6, eta_net, 1e-6)
        return P_net / (eta_safe * self.LHV_coal)

    def flue_gas_rate_kgs(self, plr, T_amb):
        m_coal = self.coal_rate_kgs(plr, T_amb)
        m_air  = m_coal * self.stoich_air * (1.0 + self.excess_air)
        return m_coal + m_air

    def flue_heat_loss_mw(self, plr, T_amb):
        T_amb   = np.asarray(T_amb, dtype=float)
        m_flue  = self.flue_gas_rate_kgs(plr, T_amb)
        T_stack = self.stack_temperature_c(plr)
        dT      = T_stack - T_amb
        return m_flue * self.cp_flue * dT / 1000.0

    def aux_power_fraction(self, plr):
        plr  = np.asarray(plr, dtype=float)
        frac = self.aux_rated_frac + self.aux_extra_frac * (1.0 - plr)
        return np.clip(frac, 0.0, 0.15)

    def power_mw(self, plr):
        return self.P_rated * np.asarray(plr, dtype=float)

    def co2_rate_kgs(self, plr, T_amb):
        return self.coal_rate_kgs(plr, T_amb) * self.CO2_per_kg_coal

    def co2_intensity_g_per_kwh(self, plr, T_amb):
        P_kw   = self.power_mw(plr) * 1e3
        co2_gs = self.co2_rate_kgs(plr, T_amb) * 1e3
        P_safe = np.where(np.asarray(P_kw) > 1e-6, P_kw, 1e-6)
        return co2_gs / P_safe * 3600.0
