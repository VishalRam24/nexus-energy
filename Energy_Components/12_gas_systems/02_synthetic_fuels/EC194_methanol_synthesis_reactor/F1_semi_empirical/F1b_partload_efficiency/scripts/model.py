"""
EC194 — Methanol Synthesis Reactor — F1b Part-Load + Thermal Model

Reaction: CO + 2H2 → CH3OH   (ΔH = -90.6 kJ/mol, exothermic)
Also: CO2 + 3H2 → CH3OH + H2O  (parallel, accounted via selectivity)

Extends conversion-efficiency (F1a) with:
  1. Part-load ratio (PLR) correction on per-pass conversion via quadratic fit.
  2. Reactor temperature drop at part-load (reduced exotherm → lower bed T).
  3. Exothermic heat recovery credit.
  4. Selectivity correction at part-load (lower T → slightly lower selectivity).

Equilibrium limited conversion (Temkin-Pyzhev approximation, semi-empirical):
    X = X_max * exp(-k_T * ((T-T_opt)/T_opt)^2) * (P/P_ref)^P_exp

References:
    Graaf, G.H. et al. (1988). Chem. Eng. Sci., 43(12), 3185-3195.
    van den Bussche, K.M. & Froment, G.F. (1996). J. Catalysis, 161(1), 1-10.
    Lurgi AG (2007). Methanol Synthesis Technology.
"""

import numpy as np


class MethanolReactorF1b:
    """Cu/ZnO/Al2O3 methanol reactor — part-load conversion + heat recovery."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.T_opt = u["T_opt"]["value"]
        self.X_max = u["X_max"]["value"]
        self.k_T = u["k_T"]["value"]
        self.P_ref = u["P_ref"]["value"]
        self.P_exp = u["P_exp"]["value"]
        self.n_CO = u["n_CO_in"]["value"]
        self.LHV_MeOH = u["LHV_MeOH"]["value"]
        self.LHV_H2 = u["LHV_H2"]["value"]
        self.DH = abs(u["DH_reaction"]["value"])
        self.f_recovery = u["f_recovery"]["value"]
        self.PLR_coeffs = u["PLR_coeffs"]["value"]
        self.selectivity_design = u["selectivity_design"]["value"]

    def _plr_factor(self, plr):
        plr = np.asarray(plr, dtype=float)
        a0, a1, a2 = self.PLR_coeffs
        f_raw = a0 + a1 * plr + a2 * plr ** 2
        f_at_1 = a0 + a1 + a2
        return np.clip(f_raw / f_at_1, 0.0, 1.0)

    def _reactor_temp(self, T_degC, plr):
        """Effective bed temperature: drops 25°C at PLR=0.3 (reduced exotherm)."""
        plr = np.asarray(plr, dtype=float)
        T = np.asarray(T_degC, dtype=float)
        return T - (1.0 - plr) * 25.0

    def conversion(self, T_reactor_degC, pressure_bar, h2_co_ratio, plr):
        """Per-pass CO conversion with PLR correction."""
        T_eff = self._reactor_temp(T_reactor_degC, plr)
        P = np.asarray(pressure_bar, dtype=float)
        r = np.asarray(h2_co_ratio, dtype=float)

        T_norm = (T_eff - self.T_opt) / self.T_opt
        X_T = self.X_max * np.exp(-self.k_T * T_norm ** 2)
        X_P = (P / self.P_ref) ** self.P_exp
        X_base = X_T * X_P

        # Stoichiometric limit: need 2 H2 per CO
        X_stoich = np.clip(r / 2.0, 0.0, 1.0)
        X_base = np.minimum(X_base, X_stoich)

        plr_f = self._plr_factor(plr)
        return np.clip(X_base * plr_f, 0.0, 1.0)

    def selectivity(self, plr):
        """CH3OH selectivity (vs DME/paraffin by-products). Drops slightly at low PLR."""
        plr = np.asarray(plr, dtype=float)
        return np.clip(self.selectivity_design * (0.85 + 0.15 * plr), 0.0, 1.0)

    def meoh_production_mol_s(self, T_reactor_degC, pressure_bar, h2_co_ratio, plr,
                               co_flow_mol_s=None):
        """Methanol production [mol/s]."""
        n = self.n_CO if co_flow_mol_s is None else np.asarray(co_flow_mol_s, dtype=float)
        X = self.conversion(T_reactor_degC, pressure_bar, h2_co_ratio, plr)
        S = self.selectivity(plr)
        return X * S * n * np.asarray(plr, dtype=float)

    def heat_recovery_kw(self, T_reactor_degC, pressure_bar, h2_co_ratio, plr,
                          co_flow_mol_s=None):
        """Recovered exothermic heat [kW]."""
        n = self.n_CO if co_flow_mol_s is None else np.asarray(co_flow_mol_s, dtype=float)
        X = self.conversion(T_reactor_degC, pressure_bar, h2_co_ratio, plr)
        return X * self.DH * n * np.asarray(plr, dtype=float) * self.f_recovery

    def overall_efficiency(self, T_reactor_degC, pressure_bar, h2_co_ratio, plr):
        """Energy efficiency: (MeOH energy out + heat credit) / H2 input energy."""
        X = self.conversion(T_reactor_degC, pressure_bar, h2_co_ratio, plr)
        S = self.selectivity(plr)
        plr_arr = np.asarray(plr, dtype=float)
        meoh_energy = X * S * self.LHV_MeOH * plr_arr
        heat_credit = X * self.DH * self.f_recovery * plr_arr
        h2_input = 2.0 * self.LHV_H2 * plr_arr
        eta = (meoh_energy + heat_credit) / (h2_input + 1e-12)
        return np.clip(eta, 0.0, 1.5)  # MeOH has higher energy density than H2 used

    def compute(self, co_flow_mol_s, h2_co_ratio, plr, T_reactor_degC=250.0,
                pressure_bar=80.0):
        """Full computation returning all outputs."""
        X = self.conversion(T_reactor_degC, pressure_bar, h2_co_ratio, plr)
        S = self.selectivity(plr)
        meoh = self.meoh_production_mol_s(T_reactor_degC, pressure_bar, h2_co_ratio,
                                           plr, co_flow_mol_s)
        Q = self.heat_recovery_kw(T_reactor_degC, pressure_bar, h2_co_ratio,
                                   plr, co_flow_mol_s)
        eta = self.overall_efficiency(T_reactor_degC, pressure_bar, h2_co_ratio, plr)
        return {
            "meoh_production_mol_s": meoh,
            "conversion": X,
            "heat_recovery_kw": Q,
            "overall_efficiency": eta,
            "selectivity": S,
        }
