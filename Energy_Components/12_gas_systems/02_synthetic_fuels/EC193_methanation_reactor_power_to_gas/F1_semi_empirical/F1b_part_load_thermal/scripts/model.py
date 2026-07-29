"""
EC193 — Methanation Reactor (Power-to-Gas) — F1b Part-Load + Thermal Model

Extends F1a equilibrium model with:
  1. Part-load ratio (PLR) correction on conversion:
     PLR_factor = a0 + a1*PLR + a2*PLR^2   (fitted quadratic, ≤1 at full load)
     At part-load, reduced feed → lower catalyst bed temperature → lower conversion.
  2. Exothermic heat recovery:
     Q_recovered = conversion * |DH_rxn| * n_CO2 * f_recovery   [kW]
  3. Selectivity: drops at part-load due to lower residence temperature.
  4. Overall efficiency: accounts for H2 input energy and heat recovery credit.

Sabatier reaction: CO2 + 4H2 -> CH4 + 2H2O  (DH = -165 kJ/mol)

Reference:
    Gao, J. et al. (2012). RSC Advances, 2, 2358-2368.
    Gotz, M. et al. (2016). Renewable Energy, 85, 1371-1390.
"""

import numpy as np


class MethanationF1b:
    """
    Catalytic methanation reactor — part-load conversion + heat recovery model.
    Builds on F1a equilibrium model adding PLR effects and thermal integration.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.T_opt   = u["T_opt"]["value"]
        self.X_max   = u["X_max"]["value"]
        self.k_T     = u["k_T"]["value"]
        self.P_ref   = u["P_ref"]["value"]
        self.P_exp   = u["P_exp"]["value"]
        self.n_CO2   = u["n_CO2_in"]["value"]
        self.LHV_CH4 = u["LHV_CH4"]["value"]
        self.LHV_H2  = u["LHV_H2"]["value"]
        self.DH      = abs(u["DH_reaction"]["value"])
        self.f_recovery = u["f_recovery"]["value"]
        self.PLR_coeffs = u["PLR_coeffs"]["value"]
        self.selectivity_design = u["selectivity_design"]["value"]
        self.catalyst_mass_kg = u["catalyst_mass_kg"]["value"]
        self.GHSV = u["GHSV"]["value"]
        self.T_design = u["T_design"]["value"]

    def _plr_factor(self, plr):
        """Part-load correction factor for conversion.
        PLR_factor = a0 + a1*PLR + a2*PLR^2
        Normalized so PLR_factor(1.0) = sum(coeffs) ≈ 1.0.
        """
        plr = np.asarray(plr, dtype=float)
        a0, a1, a2 = self.PLR_coeffs
        f_raw = a0 + a1 * plr + a2 * plr ** 2
        # Normalize so that at PLR=1.0 the factor is 1.0
        f_at_1 = a0 + a1 + a2
        return np.clip(f_raw / f_at_1, 0.0, 1.0)

    def _reactor_temp(self, T_reactor_degC, plr):
        """Effective reactor temperature accounting for part-load.
        At part-load, reduced exothermic heat generation lowers bed temperature.
        T_eff = T_reactor - (1 - PLR) * 30  (up to 30 degC drop at PLR=0.3)
        """
        plr = np.asarray(plr, dtype=float)
        T = np.asarray(T_reactor_degC, dtype=float)
        T_drop = (1.0 - plr) * 30.0
        return T - T_drop

    def conversion(self, T_reactor_degC, pressure_bar, h2_co2_ratio, plr):
        """
        CO2-to-CH4 conversion with part-load correction.

        Parameters
        ----------
        T_reactor_degC : float or array  (design reactor temperature, degC)
        pressure_bar   : float or array  (bar)
        h2_co2_ratio   : float or array  (mol/mol)
        plr            : float or array  (part-load ratio, 0.3-1.0)

        Returns
        -------
        X : float or array — conversion [0, 1]
        """
        T_eff = self._reactor_temp(T_reactor_degC, plr)
        P = np.asarray(pressure_bar, dtype=float)
        r = np.asarray(h2_co2_ratio, dtype=float)

        # Base equilibrium conversion (same as F1a)
        T_norm = (T_eff - self.T_opt) / self.T_opt
        X_T = self.X_max * np.exp(-self.k_T * T_norm ** 2)
        X_P = (P / self.P_ref) ** self.P_exp
        X_base = X_T * X_P

        # H2/CO2 stoichiometric limit
        X_stoich = np.clip(r / 4.0, 0.0, 1.0)
        X_base = np.minimum(X_base, X_stoich)

        # Part-load correction
        plr_f = self._plr_factor(plr)
        X = X_base * plr_f

        return np.clip(X, 0.0, 1.0)

    def selectivity(self, plr):
        """CH4 selectivity (vs CO side product). Drops at part-load.
        S = S_design * (0.8 + 0.2 * PLR)
        """
        plr = np.asarray(plr, dtype=float)
        return np.clip(self.selectivity_design * (0.8 + 0.2 * plr), 0.0, 1.0)

    def ch4_production(self, T_reactor_degC, pressure_bar, h2_co2_ratio, plr,
                       co2_flow_mol_s=None):
        """CH4 production rate (mol/s)."""
        if co2_flow_mol_s is None:
            co2_flow_mol_s = self.n_CO2
        n = np.asarray(co2_flow_mol_s, dtype=float)
        X = self.conversion(T_reactor_degC, pressure_bar, h2_co2_ratio, plr)
        S = self.selectivity(plr)
        return X * S * n * np.asarray(plr, dtype=float)

    def heat_recovery_kw(self, T_reactor_degC, pressure_bar, h2_co2_ratio, plr,
                         co2_flow_mol_s=None):
        """
        Recovered exothermic heat (kW).
        Q = conversion * |DH_rxn| * n_CO2 * PLR * f_recovery
        """
        if co2_flow_mol_s is None:
            co2_flow_mol_s = self.n_CO2
        n = np.asarray(co2_flow_mol_s, dtype=float)
        X = self.conversion(T_reactor_degC, pressure_bar, h2_co2_ratio, plr)
        plr_arr = np.asarray(plr, dtype=float)
        return X * self.DH * n * plr_arr * self.f_recovery

    def overall_efficiency(self, T_reactor_degC, pressure_bar, h2_co2_ratio, plr):
        """
        Overall energy efficiency including heat recovery credit.
        eta = (X * S * LHV_CH4 + Q_heat_credit) / (4 * LHV_H2)
        where Q_heat_credit = X * DH * f_recovery per mol CO2 reacted.
        """
        X = self.conversion(T_reactor_degC, pressure_bar, h2_co2_ratio, plr)
        S = self.selectivity(plr)
        plr_arr = np.asarray(plr, dtype=float)
        # Per mol CO2 fed at full load: energy balance
        ch4_energy = X * S * self.LHV_CH4 * plr_arr
        heat_credit = X * self.DH * self.f_recovery * plr_arr
        h2_input = 4.0 * self.LHV_H2 * plr_arr
        eta = (ch4_energy + heat_credit) / (h2_input + 1e-12)
        return np.clip(eta, 0.0, 1.0)

    def compute(self, co2_flow_mol_s, h2_co2_ratio, plr, T_reactor_degC=300.0,
                pressure_bar=10.0):
        """Full computation returning all outputs."""
        X = self.conversion(T_reactor_degC, pressure_bar, h2_co2_ratio, plr)
        S = self.selectivity(plr)
        ch4 = self.ch4_production(T_reactor_degC, pressure_bar, h2_co2_ratio, plr,
                                  co2_flow_mol_s)
        Q = self.heat_recovery_kw(T_reactor_degC, pressure_bar, h2_co2_ratio, plr,
                                  co2_flow_mol_s)
        eta = self.overall_efficiency(T_reactor_degC, pressure_bar, h2_co2_ratio, plr)

        return {
            "ch4_production_mol_s": ch4,
            "conversion": X,
            "heat_recovery_kw": Q,
            "overall_efficiency": eta,
            "selectivity": S,
        }
