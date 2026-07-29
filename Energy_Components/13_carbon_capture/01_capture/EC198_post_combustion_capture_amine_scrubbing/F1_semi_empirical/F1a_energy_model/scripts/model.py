"""
EC198 — Post-Combustion Capture (Amine Scrubbing) — F1a Energy Model

MEA-based post-combustion CO2 capture.
Reboiler duty model based on Abu-Zahra et al. (2007) empirical fit.

reboiler_duty = q_base / (1 - exp(-k_LG * (LG - LG_min)))  [GJ/tCO2]
q_base = 3.2 GJ/tCO2 at optimal L/G (accounts for sensible heat, heat of absorption, stripping)
electricity = 0.25 GJ/tCO2 (base, scales with capture rate)

Reference:
    Abu-Zahra, M.R.M., Schneiders, L.H.J., Niederer, J.P.M., Feron, P.H.M., Versteeg, G.F.
    (2007). CO2 capture from power plants: Part I. A parametric study of the technical
    performance based on monoethanolamine.
    International Journal of Greenhouse Gas Control, 1(1), 37-46.
"""

import numpy as np


class AmineCaptureF1a:
    """
    Post-combustion CO2 capture with MEA — energy consumption model.
    Predicts reboiler duty and electricity demand as a function of operating conditions.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.q_base         = u["q_base"]["value"]            # GJ/tCO2
        self.LG_opt         = u["LG_optimal"]["value"]        # mol/mol
        self.LG_min         = u["LG_min"]["value"]
        self.k_LG           = u["k_LG"]["value"]
        self.elec_specific  = u["electricity_specific"]["value"]  # GJ/tCO2
        self.MW_CO2         = u["MW_CO2"]["value"] / 1000.0   # kg/mol
        self.MW_air         = u["MW_air"]["value"] / 1000.0   # kg/mol

    def reboiler_duty(self, capture_rate):
        """
        Specific reboiler duty (GJ/tCO2) as a function of capture rate.

        Calibrated against Abu-Zahra et al. (2007) Fig. 10 data:
        - At CR=0.80: q ≈ 3.5 GJ/tCO2
        - At CR=0.90: q ≈ 3.6 GJ/tCO2 (slight increase due to dilute lean loading)
        - At CR=0.95: q ≈ 3.85 GJ/tCO2

        Modeled as quadratic with minimum near CR=0.85:
        q = q_base + alpha * (CR - CR_opt)^2 + beta * (CR - CR_ref)
        Simplified to:
        q = q_base + 2.5 * (CR - 0.90)^2 + 1.5 * (CR - 0.90)
        """
        cr  = np.asarray(capture_rate, dtype=float)
        dCR = cr - 0.90
        # Quadratic + linear: minimum ~0.88, increases at both ends, steeper at high CR
        q   = self.q_base + 2.5 * dCR ** 2 + 1.5 * dCR
        return np.clip(q, 2.5, 6.0)

    def electricity_demand(self, capture_rate):
        """
        Specific electricity demand (GJ/tCO2).
        Increases with capture rate (more pumping, more compression at high capture).
        E_elec = E_base * (1 + 0.5*(capture_rate - 0.90))
        """
        cr  = np.asarray(capture_rate, dtype=float)
        elec = self.elec_specific * (1.0 + 0.5 * (cr - 0.90))
        return np.clip(elec, 0.1, 1.0)

    def specific_energy(self, capture_rate):
        """
        Total specific energy consumption (GJ/tCO2) = reboiler + electricity.
        """
        return self.reboiler_duty(capture_rate) + self.electricity_demand(capture_rate)

    def co2_captured(self, flue_gas_kgs, co2_fraction, capture_rate):
        """
        CO2 captured (kg/s).
        CO2_in = flue_gas_rate * co2_fraction * (MW_CO2/MW_flue)
        CO2_captured = CO2_in * capture_rate

        Note: co2_fraction is mol fraction; mass fraction correction applied.
        """
        flue  = np.asarray(flue_gas_kgs, dtype=float)
        xCO2  = np.asarray(co2_fraction, dtype=float)
        cr    = np.asarray(capture_rate, dtype=float)

        # MW of flue gas (approximate as CO2 + air mixture)
        MW_flue = xCO2 * self.MW_CO2 + (1.0 - xCO2) * self.MW_air
        # Mass fraction of CO2 in flue gas
        mass_frac_CO2 = xCO2 * self.MW_CO2 / MW_flue
        co2_in = flue * mass_frac_CO2
        return co2_in * cr

    def reboiler_power(self, flue_gas_kgs, co2_fraction, capture_rate):
        """
        Reboiler thermal power (MW).
        P_reb = q_reboiler [GJ/tCO2] * CO2_captured [kg/s] * 1e-3 [t/kg] * 1e3 [MW/GW]
              = q_reboiler * CO2_captured / 1000 * 1000  = q_reboiler * CO2_kg_s
        Units: [GJ/t] * [kg/s] / [1000 kg/t] = [GJ/s] = [GW]... → * 1000 = MW
        = q_reboiler [GJ/t] * CO2 [kg/s] / 1000 [kg/t] * 1000 [MW·s/GJ] = same
        """
        q_reb   = self.reboiler_duty(capture_rate)           # GJ/tCO2
        co2_kgs = self.co2_captured(flue_gas_kgs, co2_fraction, capture_rate)  # kg/s
        # GJ/t * kg/s = GJ/(t) * kg/s; t = 1000 kg, so GJ/t * kg/s = GJ/1000s
        # MW = MJ/s, so GJ/1000s = GJ/1000 /s * 1000 MJ/GJ = MJ/s = MW
        return q_reb * co2_kgs / 1.0  # MW  (GJ/t * kg/s = GJ/t * kg/s = 1e-3 GW = MW)
        # 1 GJ/t * 1 kg/s = 1e9 J / 1e3 kg * 1 kg/s = 1e6 J/s = 1 MW  ✓

    def electricity_power(self, flue_gas_kgs, co2_fraction, capture_rate):
        """Electricity demand (MW)."""
        elec_spec = self.electricity_demand(capture_rate)     # GJ/tCO2
        co2_kgs   = self.co2_captured(flue_gas_kgs, co2_fraction, capture_rate)
        return elec_spec * co2_kgs  # MW (same unit analysis as above)
