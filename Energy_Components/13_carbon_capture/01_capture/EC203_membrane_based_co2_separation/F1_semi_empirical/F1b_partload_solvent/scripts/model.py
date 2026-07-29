"""
EC203 — Membrane-Based CO2 Separation — F1b Part-Load + Permeance Decline Model

Polymeric/mixed-matrix membrane CO2 separation for post-combustion and DAC.
Extends F1a with:
  1. Capture rate vs energy penalty curve: higher capture rate requires higher
     pressure ratio and/or more membrane area → exponential SEC rise.
     SEC(CR) = SEC_min + k_sec * exp(gamma * (CR - CR_ref))
  2. Membrane permeance decline with age:
     P_CO2(t) = P0_CO2 * exp(-k_age * t/8760)  [GPU units — physical aging]
     k_age ~ 0.03-0.10/year depending on membrane type.
  3. Temperature correction on permeability (Arrhenius):
     P(T) = P_ref * exp(-E_a/R * (1/T - 1/T_ref))
     Polymeric membranes: E_a,CO2 ~ -10 kJ/mol (permeability increases with T).
  4. Injection pressure effects: compression energy for CO2 permeate pressurization.
     E_comp = n * R * T / (eta * M_CO2) * ln(P_inj / P_permeate)  [kJ/kg]

Reference:
    Merkel, T.C. et al. (2010). J. Membrane Sci., 359(1-2), 126-139.
    Baker, R.W. & Low, B.T. (2014). Macromolecules, 47(20), 6999-7013.
    Favre, E. (2007). J. Membrane Sci., 294(1-2), 50-59.
"""

import numpy as np

R_GAS = 8.314  # J/(mol*K)


class MembraneF1b:
    """Membrane CO2 separation — part-load + permeance decline + temperature model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P0_CO2          = u["P0_CO2_GPU"]["value"]       # GPU (10^-10 cm3*cm/(cm2*s*cmHg))
        self.selectivity_ref = u["selectivity_CO2_N2"]["value"]  # alpha_CO2/N2
        self.k_age           = u["k_age"]["value"]             # 1/year physical aging rate
        self.E_a_kJ_mol      = u["E_a_kJ_mol"]["value"]        # activation energy (negative for CO2)
        self.T_ref_K         = u["T_ref_K"]["value"]           # K reference temperature
        self.SEC_min         = u["SEC_min"]["value"]            # kWh_e/tCO2 base electrical
        self.k_sec           = u["k_sec"]["value"]              # kWh_e/tCO2
        self.gamma_sec       = u["gamma_sec"]["value"]          # capture rate sensitivity
        self.CR_ref          = u["CR_ref"]["value"]             # reference capture rate
        self.eta_compressor  = u["eta_compressor"]["value"]     # isentropic efficiency
        self.P_feed_bar      = u["P_feed_bar"]["value"]         # feed-side pressure
        self.P_permeate_bar  = u["P_permeate_bar"]["value"]     # permeate-side pressure
        self.M_CO2           = u["M_CO2_kg_mol"]["value"]       # kg/mol
        self.gamma_gas       = u["gamma_gas"]["value"]          # Cp/Cv for CO2-rich gas

    # ------------------------------------------------------------------ #
    #  Membrane permeance factors
    # ------------------------------------------------------------------ #

    def _aging_factor(self, operating_hours):
        """Permeance decline due to physical aging (polymer chain relaxation).
        P(t) = P0 * exp(-k_age * t_years).
        """
        t_years = np.asarray(operating_hours, dtype=float) / 8760.0
        return np.exp(-self.k_age * t_years)

    def _temperature_factor(self, T_degC):
        """Arrhenius temperature correction for CO2 permeability.
        P(T) = P_ref * exp(-E_a/R * (1/T - 1/T_ref))
        CO2 in polymers: E_a < 0 → permeability increases with temperature.
        """
        T_K = np.asarray(T_degC, dtype=float) + 273.15
        exp_arg = -(self.E_a_kJ_mol * 1000.0 / R_GAS) * (1.0 / T_K - 1.0 / self.T_ref_K)
        return np.exp(exp_arg)

    def permeance_factor(self, operating_hours, T_degC):
        """Combined permeance factor (aging × temperature)."""
        return self._aging_factor(operating_hours) * self._temperature_factor(T_degC)

    # ------------------------------------------------------------------ #
    #  Capture performance
    # ------------------------------------------------------------------ #

    def capture_rate(self, co2_feed_fraction, feed_pressure_bar, operating_hours, T_degC):
        """Achievable CO2 capture rate (dimensionless) given membrane conditions.

        Uses solution-diffusion model: stage cut scales with pressure ratio and selectivity.
        stage_cut = (theta / (1 + theta)) where theta depends on P_CO2_perm/P_CO2_feed.

        Simplified: CR = 1 - (P_permeate/P_feed)^0.5 / selectivity_eff
        Capture limited by pressure ratio (Robeson upper bound trade-off).
        """
        P_f = np.asarray(feed_pressure_bar, dtype=float)
        perf = self.permeance_factor(operating_hours, T_degC)
        alpha_eff = self.selectivity_ref * np.sqrt(perf)  # selectivity improves slightly at lower T
        pressure_ratio = self.P_permeate_bar / P_f
        # Approximate stage cut = 1 - sqrt(pressure_ratio) * (1/alpha_CO2_N2)
        stage_cut = np.clip(1.0 - np.sqrt(pressure_ratio) / alpha_eff, 0.0, 0.95)
        return stage_cut

    # ------------------------------------------------------------------ #
    #  Energy calculations
    # ------------------------------------------------------------------ #

    def sec_kwh_ton(self, capture_rate):
        """Specific electrical energy consumption (kWh_e/tCO2).
        Increases exponentially with capture rate (diminishing returns).
        SEC(CR) = SEC_min + k_sec * exp(gamma * (CR - CR_ref))
        At CR=0.60: ~300 kWh/tCO2; at CR=0.90: ~700 kWh/tCO2 (Merkel 2010 range).
        """
        cr = np.asarray(capture_rate, dtype=float)
        sec = self.SEC_min + self.k_sec * np.exp(self.gamma_sec * (cr - self.CR_ref))
        return np.clip(sec, 50.0, 3000.0)

    def compression_energy_kwh_ton(self, injection_pressure_bar, T_feed_degC):
        """CO2 compression energy from permeate pressure to injection pressure (kWh/tCO2).
        Isentropic multi-stage compression (2 stages assumed):
        W = n_stages * (gamma/(gamma-1)) * (R*T/M) * [(P_out/P_in)^((gamma-1)/gamma) - 1] / eta
        """
        T_K = np.asarray(T_feed_degC, dtype=float) + 273.15
        P_in  = self.P_permeate_bar
        P_out = np.asarray(injection_pressure_bar, dtype=float)
        P_out = np.clip(P_out, P_in, None)

        g    = self.gamma_gas
        n_st = 2.0  # two compression stages
        exp_arg = (g - 1.0) / g
        PR_per_stage = (P_out / P_in) ** (1.0 / n_st)  # pressure ratio per stage
        W_J_per_mol = n_st * (g / (g - 1.0)) * R_GAS * T_K * (PR_per_stage ** exp_arg - 1.0)
        # kJ/kg = W_J_per_mol / (M_CO2 * 1000) * 1e-3
        W_kJ_kg = W_J_per_mol / (self.M_CO2 * 1e6)
        # kWh/tCO2 = kJ/kg / 3.6
        W_kwh_ton = W_kJ_kg / 3.6 / self.eta_compressor
        return np.clip(W_kwh_ton, 0.0, 500.0)

    def permeance_pct(self, operating_hours, T_degC):
        """Remaining membrane permeance (%) relative to fresh at reference T."""
        return np.clip(self.permeance_factor(operating_hours, T_degC) * 100.0, 0.0, 200.0)

    # ------------------------------------------------------------------ #
    #  Main compute
    # ------------------------------------------------------------------ #

    def compute(self, co2_feed_fraction, feed_flow_mol_s, capture_rate_target,
                T_feed_degC, operating_hours, injection_pressure_bar):
        """Full computation returning all outputs.

        Parameters
        ----------
        co2_feed_fraction       : mol/mol   — CO2 mole fraction in feed (0.04-0.15)
        feed_flow_mol_s         : mol/s     — molar feed flow rate
        capture_rate_target     : 0-1       — desired CO2 capture rate
        T_feed_degC             : degC      — membrane feed temperature
        operating_hours         : hours     — cumulative operating hours
        injection_pressure_bar  : bar       — CO2 storage injection pressure

        Returns
        -------
        dict with co2_captured_kg_h, sec_kwh_ton, compression_kwh_ton,
                  total_energy_kwh_ton, permeance_pct
        """
        cr      = np.asarray(capture_rate_target, dtype=float)
        x_CO2   = np.asarray(co2_feed_fraction, dtype=float)
        F       = np.asarray(feed_flow_mol_s, dtype=float)
        T       = np.asarray(T_feed_degC, dtype=float)

        # CO2 captured [kg/h]
        co2_mol_s   = F * x_CO2 * cr
        co2_kg_h    = co2_mol_s * self.M_CO2 * 1000.0 * 3600.0

        # Specific energy
        sec         = self.sec_kwh_ton(cr)
        e_comp      = self.compression_energy_kwh_ton(injection_pressure_bar, T)
        e_total     = sec + e_comp

        # Permeance remaining
        perm_pct    = self.permeance_pct(operating_hours, T)

        return {
            "co2_captured_kg_h":      co2_kg_h,
            "sec_kwh_ton":            sec,
            "compression_kwh_ton":    e_comp,
            "total_energy_kwh_ton":   e_total,
            "permeance_pct":          perm_pct,
        }
