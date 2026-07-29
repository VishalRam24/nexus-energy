"""
EC118 — Small Modular Reactor (SMR) — F1a Steady-State Power Map

Integral PWR-type SMR (50-300 MWe class, e.g. NuScale VOYGR). Designed for deeper
load-following than traditional large LWRs (PLR_min ~ 0.20).

Model equations:
    P_thermal  = P_thermal_rated * PLR                              [MW_th]
    f_PLR      = 1.0 for PLR >= PLR_derate
               = PLR / PLR_derate  for PLR < PLR_derate
    P_electric = P_thermal * eta_cycle * eta_gen * f_PLR            [MW_e]
    eta_net    = P_electric / P_thermal                             [-]
    dT         = P_thermal*1e6 / (m_dot * cp * 1000)                [degC]
    T_outlet   = T_inlet + dT

Reference:
    IAEA (2022). Advances in Small Modular Reactor Technology Developments.
    NuScale Power Module Design Certification Documents (2020).
"""

import numpy as np


class SMRF1a:
    """Small Modular Reactor — steady-state power map model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_thermal_rated = u["P_thermal_mw"]["value"]
        self.P_electric_rated = u["P_electric_mw"]["value"]
        self.eta_cycle = u["eta_cycle"]["value"]
        self.eta_gen = u["eta_gen"]["value"]
        self.T_inlet = u["T_inlet_c"]["value"]
        self.T_outlet_rated = u["T_outlet_c"]["value"]
        self.PLR_min = u["PLR_min"]["value"]
        self.cp = u["cp_coolant_kjkgk"]["value"]
        self.m_dot_nominal = u["m_dot_nominal_kgs"]["value"]
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

    def coolant_outlet_temp(self, part_load_ratio, coolant_flow_kgs=None):
        """Primary coolant hot-leg temperature [degC]."""
        PLR = np.clip(np.asarray(part_load_ratio, dtype=float), self.PLR_min, 1.0)
        P_th_W = self.thermal_power(PLR) * 1e6
        m_dot = self.m_dot_nominal if coolant_flow_kgs is None else np.asarray(coolant_flow_kgs, dtype=float)
        cp_J = self.cp * 1000.0
        dT = P_th_W / (m_dot * cp_J)
        return self.T_inlet + dT
