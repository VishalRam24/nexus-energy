"""
EC117 — Boiling Water Reactor (BWR) — F1a Steady-State Power Map

Direct-cycle reactor — coolant boils inside the core, steam goes directly to the turbine
(no separate steam generator as in PWR).

Model equations:
    P_thermal  = P_thermal_rated * PLR                              [MW_th]
    f_PLR      = 1.0 for PLR >= PLR_derate
               = PLR / PLR_derate  for PLR < PLR_derate
    P_electric = P_thermal * eta_cycle * eta_gen * f_PLR            [MW_e]
    eta_net    = P_electric / P_thermal                             [-]
    Q_subcool  = m_dot * cp * (T_sat - T_feedwater)                 (subcooling preheat)
    Q_evap     = P_thermal*1e3 - Q_subcool                          [kW] -> evaporation
    m_dot      = Q_evap / h_fg                                      [kg/s]

Reference:
    Todreas, N.E. & Kazimi, M.S. (2012). Nuclear Systems, 2nd Edition. CRC Press.
    GE BWR/6 reference plant data.
"""

import numpy as np


class BWRF1a:
    """Boiling Water Reactor — steady-state power map model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_thermal_rated = u["P_thermal_mw"]["value"]
        self.P_electric_rated = u["P_electric_mw"]["value"]
        self.eta_cycle = u["eta_cycle"]["value"]
        self.eta_gen = u["eta_gen"]["value"]
        self.T_feedwater = u["T_feedwater_c"]["value"]
        self.T_steam = u["T_steam_c"]["value"]
        self.PLR_min = u["PLR_min"]["value"]
        self.h_fg = u["h_fg_kjkg"]["value"]
        self.cp = u["cp_water_kjkgk"]["value"]
        self.m_dot_nominal = u["m_dot_steam_nominal_kgs"]["value"]
        self.PLR_derate = u["PLR_linear_derate_below"]["value"]

    def _f_plr(self, PLR):
        PLR = np.asarray(PLR, dtype=float)
        return np.where(PLR >= self.PLR_derate, 1.0, PLR / self.PLR_derate)

    def thermal_power(self, part_load_ratio):
        """Reactor thermal power [MW_th]."""
        PLR = np.clip(np.asarray(part_load_ratio, dtype=float), self.PLR_min, 1.0)
        return self.P_thermal_rated * PLR

    def electric_power(self, part_load_ratio):
        """Net electrical output [MW_e]."""
        PLR = np.clip(np.asarray(part_load_ratio, dtype=float), self.PLR_min, 1.0)
        P_th = self.thermal_power(PLR)
        f = self._f_plr(PLR)
        return P_th * self.eta_cycle * self.eta_gen * f

    def efficiency(self, part_load_ratio):
        """Net thermal efficiency [-]."""
        PLR = np.clip(np.asarray(part_load_ratio, dtype=float), self.PLR_min, 1.0)
        P_th = self.thermal_power(PLR)
        P_el = self.electric_power(PLR)
        return np.where(P_th > 0, P_el / P_th, 0.0)

    def steam_mass_flow(self, part_load_ratio):
        """
        Steam mass flow [kg/s] computed from energy balance:
        Q_in = m_dot * (cp*(T_sat - T_fw) + h_fg)
        """
        PLR = np.clip(np.asarray(part_load_ratio, dtype=float), self.PLR_min, 1.0)
        P_th_kw = self.thermal_power(PLR) * 1e3                  # kW
        h_per_kg = self.cp * (self.T_steam - self.T_feedwater) + self.h_fg
        return P_th_kw / h_per_kg
