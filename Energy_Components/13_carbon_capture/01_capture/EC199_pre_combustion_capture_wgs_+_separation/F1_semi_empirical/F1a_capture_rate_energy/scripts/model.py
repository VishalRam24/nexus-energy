"""
EC199 — Pre-Combustion Capture (WGS + Separation) — F1a Capture Rate + Energy Model

Pre-combustion capture process:
  1. Fuel (syngas: CO + H2) shifted via Water-Gas Shift reactor:
         CO + H2O → CO2 + H2   (ΔH = -41.1 kJ/mol, exothermic, 200-400 degC)
  2. CO2 separated from H2-rich stream by physical solvent (Selexol/Rectisol) or PSA/membrane.
  3. Decarbonized H2 fuel fed to gas turbine or fuel cell.

F1a models:
  (a) WGS conversion fraction:
        X_WGS = X_max * exp(-k_T * ((T-T_opt)/T_opt)^2) * (P/P_ref)^(-P_exp)
        (Lower P favors WGS equilibrium slightly; temperature window 200-400 degC)
  (b) Separation energy:
        E_sep = E_base * exp(-k_sep * (CO2_partial / P_ref_sep))  [GJ/tCO2]
        Lower CO2 partial pressure → worse separation → more energy.
  (c) CO2 capture rate from syngas: CR = X_WGS * eta_sep
  (d) H2 product purity and yield.

References:
    IEAGHG (2014). CO2 Capture at Gas Fired Power Plants. Report 2012/8.
    IPCC (2005). Special Report on Carbon Dioxide Capture and Storage. Ch. 3.
    Lozza, G. & Chiesa, P. (2002). J. Eng. Gas Turbines Power, 124(1), 82-88.
    Kunze, C. & Spliethoff, H. (2012). Applied Energy, 94, 109-116.
    DOE/NETL-2010/1397 (2010). Cost and Performance Baseline for Fossil Energy Plants.
"""

import numpy as np


class PreCombustionCaptureF1a:
    """
    Pre-combustion CO2 capture: WGS conversion + physical solvent separation.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.T_WGS_opt     = u["T_WGS_opt"]["value"]          # degC
        self.X_WGS_max     = u["X_WGS_max"]["value"]
        self.k_T_WGS       = u["k_T_WGS"]["value"]
        self.P_ref_WGS     = u["P_ref_WGS"]["value"]          # bar
        self.P_exp_WGS     = u["P_exp_WGS"]["value"]
        self.eta_sep_design = u["eta_sep_design"]["value"]     # separation efficiency
        self.E_sep_base    = u["E_sep_base"]["value"]          # GJ/tCO2
        self.k_sep         = u["k_sep"]["value"]              # separation energy sensitivity
        self.P_ref_sep_bar = u["P_ref_sep_bar"]["value"]      # bar CO2 partial pressure ref
        self.E_compression = u["E_compression"]["value"]      # GJ/tCO2
        self.MW_CO2        = u["MW_CO2"]["value"] / 1000.0    # kg/mol
        self.MW_H2         = u["MW_H2"]["value"] / 1000.0    # kg/mol
        self.MW_CO         = u["MW_CO"]["value"] / 1000.0    # kg/mol
        self.DH_WGS        = abs(u["DH_WGS"]["value"])        # kJ/mol (exothermic)

    def wgs_conversion(self, T_degC, P_bar, steam_co_ratio=3.0):
        """
        WGS CO conversion fraction.
        X = X_max * exp(-k_T*((T-T_opt)/T_opt)^2) * (P/P_ref)^(-P_exp)
        S/C ratio correction: excess steam shifts equilibrium right.
        """
        T = np.asarray(T_degC, dtype=float)
        P = np.asarray(P_bar, dtype=float)
        sc = np.asarray(steam_co_ratio, dtype=float)

        T_norm = (T - self.T_WGS_opt) / self.T_WGS_opt
        X_T = self.X_WGS_max * np.exp(-self.k_T_WGS * T_norm ** 2)
        X_P = (P / self.P_ref_WGS) ** (-self.P_exp_WGS)  # higher P slightly reduces X

        # Steam-to-CO correction: extra steam drives equilibrium right
        sc_factor = np.clip(1.0 + 0.05 * (sc - 3.0), 0.9, 1.1)
        return np.clip(X_T * X_P * sc_factor, 0.0, 1.0)

    def separation_energy_GJ_tCO2(self, CO2_partial_bar):
        """
        Specific separation energy [GJ/tCO2].
        Physical solvents (Selexol): E decreases with higher CO2 partial pressure.
        E_sep = E_base * exp(-k_sep * P_CO2/P_ref_sep)
        """
        P_CO2 = np.asarray(CO2_partial_bar, dtype=float)
        return np.clip(
            self.E_sep_base * np.exp(-self.k_sep * P_CO2 / self.P_ref_sep_bar),
            0.1, 2.0
        )

    def total_energy_GJ_tCO2(self, T_degC, P_bar, co2_mole_fraction=0.4):
        """Total energy penalty per tCO2 = separation + compression."""
        CO2_partial = np.asarray(P_bar, dtype=float) * np.asarray(co2_mole_fraction, dtype=float)
        E_sep = self.separation_energy_GJ_tCO2(CO2_partial)
        return E_sep + self.E_compression

    def capture_rate(self, T_degC, P_bar, steam_co_ratio=3.0):
        """Overall CO2 capture rate = WGS conversion × separation efficiency."""
        X = self.wgs_conversion(T_degC, P_bar, steam_co_ratio)
        return np.clip(X * self.eta_sep_design, 0.0, 1.0)

    def co2_captured_kg_s(self, syngas_flow_mol_s, co_fraction,
                           T_degC, P_bar, steam_co_ratio=3.0):
        """CO2 captured [kg/s]."""
        n = np.asarray(syngas_flow_mol_s, dtype=float)
        x_CO = np.asarray(co_fraction, dtype=float)
        X_WGS = self.wgs_conversion(T_degC, P_bar, steam_co_ratio)
        # CO converted to CO2 by WGS
        n_CO2_mol_s = n * x_CO * X_WGS * self.eta_sep_design
        return n_CO2_mol_s * self.MW_CO2

    def h2_yield_mol_s(self, syngas_flow_mol_s, co_fraction, h2_fraction,
                        T_degC, P_bar, steam_co_ratio=3.0):
        """H2 product [mol/s] = original H2 + H2 from WGS."""
        n = np.asarray(syngas_flow_mol_s, dtype=float)
        X = self.wgs_conversion(T_degC, P_bar, steam_co_ratio)
        n_H2_orig = n * np.asarray(h2_fraction, dtype=float)
        n_H2_WGS  = n * np.asarray(co_fraction, dtype=float) * X
        return n_H2_orig + n_H2_WGS

    def wgs_heat_kW(self, syngas_flow_mol_s, co_fraction, T_degC, P_bar):
        """Exothermic WGS heat released [kW]."""
        n = np.asarray(syngas_flow_mol_s, dtype=float)
        x_CO = np.asarray(co_fraction, dtype=float)
        X = self.wgs_conversion(T_degC, P_bar)
        return n * x_CO * X * self.DH_WGS   # kJ/s = kW

    def compute(self, syngas_flow_mol_s, co_fraction, h2_fraction,
                T_WGS_C, P_bar, steam_co_ratio=3.0):
        """Full computation."""
        X_WGS = self.wgs_conversion(T_WGS_C, P_bar, steam_co_ratio)
        CR    = self.capture_rate(T_WGS_C, P_bar, steam_co_ratio)
        co2_kg_s = self.co2_captured_kg_s(syngas_flow_mol_s, co_fraction, T_WGS_C, P_bar)
        h2_mol_s = self.h2_yield_mol_s(syngas_flow_mol_s, co_fraction, h2_fraction, T_WGS_C, P_bar)
        co2_partial = P_bar * co_fraction * X_WGS  # approx CO2 partial pressure in product
        E_total = self.total_energy_GJ_tCO2(T_WGS_C, P_bar, co_fraction * X_WGS)
        Q_WGS = self.wgs_heat_kW(syngas_flow_mol_s, co_fraction, T_WGS_C, P_bar)

        return {
            "wgs_conversion": X_WGS,
            "capture_rate": CR,
            "co2_captured_kg_s": co2_kg_s,
            "h2_yield_mol_s": h2_mol_s,
            "total_energy_GJ_tCO2": E_total,
            "wgs_heat_kW": Q_WGS,
        }
