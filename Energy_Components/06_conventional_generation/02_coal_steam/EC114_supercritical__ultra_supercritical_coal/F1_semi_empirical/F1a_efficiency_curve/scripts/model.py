"""
EC114 — Supercritical / Ultra-Supercritical Coal Plant — F1a Efficiency Curve

Net electrical efficiency as a function of part-load ratio and ambient temperature:
    eta(PLR, T_amb) = eta_iso * f_PLR(PLR) * f_amb(T_amb)
    f_PLR(PLR) = a0 + a1*PLR + a2*PLR^2    (part-load correction)
    f_amb(T_amb) = 1 - k_amb*(T_amb - T_ref) (condenser back-pressure derating)

Technology levels:
    SC  (Supercritical):        ~600 C / 250 bar; eta_net ~42-44%
    USC (Ultra-Supercritical):  ~700 C / 300 bar; eta_net ~44-47%

Derived outputs:
    P_out        = P_rated * PLR                           [MW_e]
    coal_rate    = P_out / (eta * LHV_coal)                [kg/s]
    CO2_rate     = coal_rate * CO2_per_kg_coal             [kg/s]
    CO2_intensity= CO2_rate / P_out * 3600                 [g/kWh]  (~750-850 g/kWh)

References:
    Weitzel, P.S. (2011). Steam generator for advanced ultra supercritical power
    plants. ASME PVP-2011-57934.
    IEA Clean Coal Centre, CCC/168 (2010). Efficiency and emissions performance of
    new and upgrading coal-fired power plants.
    Luo, Z., Zhu, S., Wang, R., Xu, G. (2013). Analysis and optimization of
    combined cycles with SC/USC steam parameters. Energy, 57, 236-244.
"""

import numpy as np


class SupercriticalCoalF1a:
    """Supercritical / Ultra-Supercritical coal plant — efficiency curve semi-empirical model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated         = u["rated_power_mw"]["value"]       # MW_e
        self.eta_iso         = u["eta_iso"]["value"]              # dimensionless
        self.T_amb_ref       = u["T_amb_ref"]["value"]            # degC
        self.k_amb           = u["k_amb"]["value"]                # 1/K
        self.a0              = u["plr_coeffs"]["a0"]["value"]
        self.a1              = u["plr_coeffs"]["a1"]["value"]
        self.a2              = u["plr_coeffs"]["a2"]["value"]
        self.LHV_coal        = u["LHV_coal"]["value"]             # MJ/kg
        self.CO2_per_kg_coal = u["CO2_per_kg_coal"]["value"]      # kg_CO2/kg_coal
        self.steam_temp_c    = u["steam_temp_c"]["value"]         # degC
        self.technology      = u.get("technology", "USC")

    # ------------------------------------------------------------------
    # Correction factors
    # ------------------------------------------------------------------

    def f_plr(self, plr):
        """Quadratic part-load efficiency correction factor."""
        plr = np.asarray(plr, dtype=float)
        return self.a0 + self.a1 * plr + self.a2 * plr ** 2

    def f_amb(self, T_amb):
        """Linear ambient-temperature efficiency derating (condenser back-pressure)."""
        T_amb = np.asarray(T_amb, dtype=float)
        return 1.0 - self.k_amb * (T_amb - self.T_amb_ref)

    # ------------------------------------------------------------------
    # Primary outputs
    # ------------------------------------------------------------------

    def efficiency(self, plr, T_amb):
        """Net LHV efficiency (dimensionless)."""
        return self.eta_iso * self.f_plr(plr) * self.f_amb(T_amb)

    def power_mw(self, plr):
        """Electrical output power [MW_e]."""
        return self.P_rated * np.asarray(plr, dtype=float)

    def coal_rate_kgs(self, plr, T_amb):
        """Coal mass flow rate [kg/s]."""
        P_out    = self.power_mw(plr)                     # MW_e
        eta      = self.efficiency(plr, T_amb)
        eta_safe = np.where(np.asarray(eta) > 1e-6, eta, 1e-6)
        fuel_mw  = P_out / eta_safe                       # MW_th
        return fuel_mw / self.LHV_coal                    # kg/s

    def co2_rate_kgs(self, plr, T_amb):
        """CO2 emission rate [kg/s]."""
        return self.coal_rate_kgs(plr, T_amb) * self.CO2_per_kg_coal

    def co2_intensity_g_per_kwh(self, plr, T_amb):
        """
        CO2 emission intensity [g_CO2/kWh_e].
        Typical: 750-850 g/kWh for SC/USC coal (lower than subcritical).
        """
        P_out_kw = self.power_mw(plr) * 1e3
        co2_gs   = self.co2_rate_kgs(plr, T_amb) * 1e3    # g/s
        P_safe   = np.where(np.asarray(P_out_kw) > 1e-6, P_out_kw, 1e-6)
        return co2_gs / P_safe * 3600.0

    def carnot_efficiency(self, T_cond_c=40.0):
        """Carnot efficiency for steam at self.steam_temp_c vs condenser T_cond_c [degC]."""
        T_H = self.steam_temp_c + 273.15
        T_C = T_cond_c + 273.15
        return 1.0 - T_C / T_H
