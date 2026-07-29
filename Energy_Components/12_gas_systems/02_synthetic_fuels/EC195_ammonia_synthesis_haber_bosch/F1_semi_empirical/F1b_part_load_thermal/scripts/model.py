"""
EC195 — Ammonia Synthesis (Haber-Bosch) — F1b Part-Load + Thermal Model

Extends F1a per-pass conversion model with:
  1. Part-load ratio effect on pressure and conversion:
     At part-load, compressor operates off-design, effective pressure drops.
     P_eff = P_design * (0.85 + 0.15 * PLR)
  2. Recycle ratio: R = 1/X_single_pass - 1 (more recycle at lower conversion)
  3. Compression energy: multi-stage isentropic with intercooling
     W_comp = n_stages * (gamma/(gamma-1)) * R_gas * T_in * ((P_out/P_in)^((gamma-1)/(n*gamma)) - 1) / eta
  4. Purge fraction: increases at part-load to maintain loop purity
  5. Energy per ton NH3: compression + heating + recycle penalties

N2 + 3H2 -> 2NH3  (exothermic, DH = -92 kJ/mol N2)

Reference:
    Appl, M. (2011). Ammonia. In Ullmann's Encyclopedia of Industrial Chemistry.
    Patil, A. et al. (2015). Procedia Engineering, 138, 229-236.
"""

import numpy as np


class AmmoniaF1b:
    """
    Haber-Bosch ammonia synthesis — part-load model with recycle loop and energy balance.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.T_ref      = u["T_ref"]["value"] + 273.15
        self.P_ref      = u["P_ref"]["value"]
        self.X_ref      = u["X_ref"]["value"]
        self.E_a_R      = u["E_a_R"]["value"]
        self.P_exp      = u["P_exp"]["value"]
        self.E_specific = u["E_specific"]["value"]
        self.MW_NH3     = u["MW_NH3"]["value"] / 1000.0
        self.n_N2       = u["n_N2_in"]["value"]
        self.conversion_design = u["conversion_design"]["value"]
        self.compression_stages = u["compression_stages"]["value"]
        self.eta_compressor = u["eta_compressor"]["value"]
        self.purge_design = u["purge_fraction_design"]["value"]
        self.gamma = u["gamma"]["value"]
        self.T_inlet_K = u["T_inlet_K"]["value"]
        self.R_gas = u["R_gas"]["value"]
        self.P_design = u["P_design"]["value"]
        self.T_design = u["T_design"]["value"]

    def _effective_pressure(self, pressure_bar, plr):
        """At part-load, compressor delivers less pressure."""
        P = np.asarray(pressure_bar, dtype=float)
        plr = np.asarray(plr, dtype=float)
        return P * (0.85 + 0.15 * plr)

    def single_pass_conversion(self, temperature_c, pressure_bar, plr):
        """
        Single-pass conversion with part-load pressure effect.
        X = X_ref * (P_eff/P_ref)^0.5 * exp(-E_a/R * (1/T - 1/T_ref))
        """
        T = np.asarray(temperature_c, dtype=float) + 273.15
        P_eff = self._effective_pressure(pressure_bar, plr)
        X = (self.X_ref
             * (P_eff / self.P_ref) ** self.P_exp
             * np.exp(-self.E_a_R * (1.0 / T - 1.0 / self.T_ref)))
        return np.clip(X, 0.01, 0.5)

    def recycle_ratio(self, temperature_c, pressure_bar, plr):
        """Recycle ratio = 1/X_sp - 1. More recycle at lower conversion."""
        X_sp = self.single_pass_conversion(temperature_c, pressure_bar, plr)
        return 1.0 / X_sp - 1.0

    def purge_fraction(self, plr):
        """Purge fraction increases at part-load to control inert buildup.
        purge = purge_design * (1 + 0.5 * (1 - PLR))
        """
        plr = np.asarray(plr, dtype=float)
        return self.purge_design * (1.0 + 0.5 * (1.0 - plr))

    def nh3_production(self, temperature_c, pressure_bar, plr,
                       n2_flow_mol_s=None, h2_n2_ratio=3.0):
        """NH3 production rate (mol/s).
        Overall conversion accounts for recycle: nearly all N2 converted minus purge loss.
        NH3 = 2 * n_N2 * PLR * (1 - purge) * X_overall
        where X_overall ~ 1 - purge (at steady state with recycle)
        """
        if n2_flow_mol_s is None:
            n2_flow_mol_s = self.n_N2
        n = np.asarray(n2_flow_mol_s, dtype=float)
        plr = np.asarray(plr, dtype=float)
        purge = self.purge_fraction(plr)
        # Overall loop conversion: everything that doesn't get purged
        X_overall = 1.0 - purge
        return 2.0 * n * plr * X_overall

    def compression_energy_kj_per_mol(self, pressure_bar, plr):
        """Multi-stage isentropic compression energy (kJ/mol syngas).
        W = n_stages * (gamma/(gamma-1)) * R * T_in * ((P_out/P_in)^((gamma-1)/(n*gamma)) - 1) / eta
        Feed comes at ~30 bar (after primary compression), final pressure = P_eff.
        """
        P_eff = self._effective_pressure(pressure_bar, plr)
        P_in = 30.0  # bar, typical feed pressure
        n_s = self.compression_stages
        g = self.gamma

        # Pressure ratio per stage
        r_per_stage = (P_eff / P_in) ** (1.0 / n_s)
        # Work per stage per mol
        exp_term = (g - 1.0) / g
        W_per_stage = (g / (g - 1.0)) * self.R_gas * self.T_inlet_K * (
            r_per_stage ** exp_term - 1.0) / self.eta_compressor
        # Total work in kJ/mol
        W_total = n_s * W_per_stage / 1000.0
        return W_total

    def energy_kwh_per_ton(self, temperature_c, pressure_bar, plr):
        """Specific energy consumption (kWh/ton NH3).
        Uses semi-empirical approach: E_design scaled by conversion ratio.
        Lower conversion → more recycle → higher energy.
        E = E_design_kwh * (X_design / X_actual)^0.3
        E_design ~ 28 GJ/t = 7778 kWh/t
        """
        X_sp = self.single_pass_conversion(temperature_c, pressure_bar, plr)
        X_design = self.conversion_design

        # Base energy at design (GJ/t -> kWh/t: *1e6/3600 = *277.78)
        E_design_kwh = self.E_specific * 277.78  # ~7778 kWh/t

        # Scale: lower conversion → more passes → more energy
        E = E_design_kwh * (X_design / np.clip(X_sp, 0.01, 1.0)) ** 0.3

        # Additional penalty for recycle compressor at part-load
        plr_arr = np.asarray(plr, dtype=float)
        plr_penalty = 1.0 + 0.1 * (1.0 - plr_arr)

        return np.clip(E * plr_penalty, 5000.0, 20000.0)

    def compute(self, n2_flow_mol_s, h2_n2_ratio, plr, pressure_bar=200.0,
                temperature_c=450.0):
        """Full computation returning all outputs."""
        X_sp = self.single_pass_conversion(temperature_c, pressure_bar, plr)
        R_loop = self.recycle_ratio(temperature_c, pressure_bar, plr)
        nh3 = self.nh3_production(temperature_c, pressure_bar, plr, n2_flow_mol_s,
                                  h2_n2_ratio)
        E = self.energy_kwh_per_ton(temperature_c, pressure_bar, plr)
        purge = self.purge_fraction(plr)

        return {
            "nh3_production_mol_s": nh3,
            "single_pass_conversion": X_sp,
            "recycle_ratio": R_loop,
            "energy_kwh_per_ton": E,
            "purge_fraction": purge,
        }
