"""
EC202 — Direct Air Capture (DAC) Liquid Solvent — F1b Part-Load + Solvent Degradation Model

KOH/Ca(OH)2 liquid solvent DAC (Carbon Engineering / 1PointFive style).
Extends F1a with:
  1. Part-load contactors: fewer air contactors online → proportional capture,
     but specific energy penalty at PLR < 1 (fixed regeneration infrastructure).
  2. Solvent carbonate build-up (degradation): K2CO3 accumulates, reducing KOH activity.
     capacity_factor = exp(-k_deg * operating_hours / 8760)
     k_deg ~ 0.05/year (5% capacity loss per year due to carbonate poisoning).
  3. Calciner temperature effect on regeneration energy:
     Q_calciner(T) = Q_base * (1 + alpha * (T - T_ref) / T_ref)
     Lower calciner temperature → higher specific energy (incomplete regeneration).
  4. Injection pressure effects on CO2 liquefaction energy:
     E_liq(P) = 0.25 * ln(P/P_atm) [GJ/tCO2] — compression work scales logarithmically.

Physics references:
    Keith, D.W. et al. (2018). Joule, 2(8), 1573-1594.
    McQueen, N. et al. (2021). Progress in Energy, 3(3), 032001.
    Fasihi, M. et al. (2019). J. Cleaner Production, 224, 957-980.
"""

import numpy as np


class DACLSF1b:
    """DAC Liquid Solvent — part-load + solvent degradation + pressure effects model."""

    def __init__(self, params: dict):
        u = params["unit"]
        # Base energy parameters
        self.Q_calciner_base = u["Q_calciner_base"]["value"]   # GJ/tCO2 thermal
        self.E_el_base       = u["E_el_base"]["value"]         # kWh_e/tCO2 electrical
        self.T_calciner_ref  = u["T_calciner_ref"]["value"]    # degC reference
        self.alpha_calciner  = u["alpha_calciner"]["value"]    # sensitivity coeff
        # Capture parameters
        self.k_cap_koh       = u["k_cap_koh"]["value"]         # mol_CO2/(m2*h) contactor capacity
        self.area_contactor  = u["area_contactor_m2"]["value"] # m2
        self.n_contactors    = int(u["n_contactors"]["value"])  # total contactors
        self.capture_eff     = u["capture_efficiency"]["value"]
        self.CO2_ppm         = u["CO2_concentration_ppm"]["value"]
        self.rho_air         = u["rho_air"]["value"]
        self.M_CO2           = u["CO2_molar_mass"]["value"]     # g/mol
        self.M_air           = u["air_molar_mass"]["value"]
        # Degradation
        self.k_deg           = u["k_deg"]["value"]              # 1/year
        # Injection pressure energy
        self.E_liq_coeff     = u["E_liq_coeff"]["value"]        # GJ/tCO2 per ln(P/P_atm)
        self.P_atm           = u["P_atm"]["value"]              # bar

        # CO2 concentration in air [kg_CO2/m3_air]
        self.CO2_conc_kgm3 = (self.CO2_ppm * 1e-6) * (self.M_CO2 / self.M_air) * self.rho_air

    # ------------------------------------------------------------------ #
    #  Core degradation / modifying factors
    # ------------------------------------------------------------------ #

    def _capacity_factor(self, operating_hours):
        """Remaining KOH activity factor.
        Exponential decay due to K2CO3 build-up: f = exp(-k_deg * t_years).
        """
        t_years = np.asarray(operating_hours, dtype=float) / 8760.0
        return np.exp(-self.k_deg * t_years)

    def _plr_energy_factor(self, plr):
        """Part-load energy penalty factor.
        At PLR < 1: fewer contactors but fixed calciner → specific energy rises.
        factor = 0.7 + 0.3/PLR  (at PLR=1: 1.0, at PLR=0.5: 1.3, at PLR=0.3: ~1.7)
        Clipped at 2.5 to avoid unrealistic extremes.
        """
        plr = np.asarray(plr, dtype=float)
        return np.clip(0.7 + 0.3 / (plr + 1e-9), 1.0, 2.5)

    def _calciner_energy_factor(self, T_calciner_degC):
        """Calciner temperature deviation factor.
        Lower T → incomplete CaCO3 decomposition → higher specific heat.
        factor = 1 + alpha * (T_ref - T) / T_ref   [> 1 when T < T_ref]
        """
        T = np.asarray(T_calciner_degC, dtype=float)
        return np.clip(1.0 + self.alpha_calciner * (self.T_calciner_ref - T) / self.T_calciner_ref,
                       0.8, 2.0)

    # ------------------------------------------------------------------ #
    #  Energy calculations
    # ------------------------------------------------------------------ #

    def calciner_duty_gj_ton(self, T_calciner_degC, plr, operating_hours):
        """Calciner thermal duty (GJ/tCO2).
        Q = Q_base * T_factor * PLR_factor * (1/capacity_factor)
        """
        cap_f  = self._capacity_factor(operating_hours)
        T_f    = self._calciner_energy_factor(T_calciner_degC)
        plr_f  = self._plr_energy_factor(plr)
        Q = self.Q_calciner_base * T_f * plr_f / (cap_f + 1e-9)
        return np.clip(Q, 5.0, 25.0)

    def electrical_kwh_ton(self, plr, operating_hours):
        """Electrical energy consumption (kWh_e/tCO2).
        Fans (contactor fans) + pumps scale with PLR and degradation.
        E = E_base * (0.6 + 0.4/PLR) / capacity_factor
        """
        plr   = np.asarray(plr, dtype=float)
        cap_f = self._capacity_factor(operating_hours)
        fan_pump_f = np.clip(0.6 + 0.4 / (plr + 1e-9), 1.0, 2.5)
        E = self.E_el_base * fan_pump_f / (cap_f + 1e-9)
        return np.clip(E, 200.0, 2000.0)

    def liquefaction_energy_gj_ton(self, injection_pressure_bar):
        """CO2 compression/liquefaction energy (GJ/tCO2).
        E_liq = E_liq_coeff * ln(P / P_atm)
        At P=150 bar: E_liq ~ 0.25 * ln(150) = 1.24 GJ/tCO2
        """
        P = np.asarray(injection_pressure_bar, dtype=float)
        P = np.clip(P, self.P_atm, None)
        return self.E_liq_coeff * np.log(P / self.P_atm)

    def total_specific_energy_gj_ton(self, T_calciner_degC, plr, operating_hours,
                                     injection_pressure_bar):
        """Total specific energy (GJ/tCO2): calciner + electrical + compression.
        Electrical converted: kWh/tCO2 / 277.78 = GJ/tCO2.
        """
        Q_calc  = self.calciner_duty_gj_ton(T_calciner_degC, plr, operating_hours)
        E_el_gj = self.electrical_kwh_ton(plr, operating_hours) / 277.78
        E_liq   = self.liquefaction_energy_gj_ton(injection_pressure_bar)
        return Q_calc + E_el_gj + E_liq

    # ------------------------------------------------------------------ #
    #  Capture rate
    # ------------------------------------------------------------------ #

    def capture_rate_kgh(self, air_flow_m3h, plr, operating_hours):
        """CO2 capture rate (kg/h).
        Limited by min(air-side CO2, KOH contactor capacity * PLR * capacity_factor).
        """
        flow      = np.asarray(air_flow_m3h, dtype=float)
        plr_arr   = np.asarray(plr, dtype=float)
        cap_f     = self._capacity_factor(operating_hours)

        # Air-side available CO2 (kg/h)
        co2_air_kgh = flow * self.CO2_conc_kgm3 * self.capture_eff

        # KOH contactor capacity (kg/h)
        # k_cap [mol_CO2/(m2*h)] * area * n_contactors_online * M_CO2/1000
        n_online        = self.n_contactors * plr_arr
        co2_cap_mol_h   = self.k_cap_koh * self.area_contactor * n_online * cap_f
        co2_cap_kgh     = co2_cap_mol_h * self.M_CO2 / 1000.0

        return np.minimum(co2_air_kgh, co2_cap_kgh)

    def solvent_capacity_pct(self, operating_hours):
        """Remaining KOH absorptive capacity (%).
        100 * exp(-k_deg * t_years).
        """
        return self._capacity_factor(operating_hours) * 100.0

    # ------------------------------------------------------------------ #
    #  Main compute
    # ------------------------------------------------------------------ #

    def compute(self, air_flow_m3h, T_calciner_degC, plr, operating_hours,
                injection_pressure_bar):
        """Full computation returning all outputs.

        Parameters
        ----------
        air_flow_m3h            : m3/h  — air throughput through contactors
        T_calciner_degC         : degC  — calciner operating temperature (750–950 C)
        plr                     : 0–1   — part-load ratio (fraction of contactors online)
        operating_hours         : h     — cumulative operating hours since solvent refresh
        injection_pressure_bar  : bar   — CO2 injection/storage pressure (1–300 bar)

        Returns
        -------
        dict with capture_rate_kgh, calciner_duty_gj_ton, electrical_kwh_ton,
                  liquefaction_energy_gj_ton, total_specific_energy_gj_ton,
                  solvent_capacity_pct
        """
        co2_kgh  = self.capture_rate_kgh(air_flow_m3h, plr, operating_hours)
        Q_calc   = self.calciner_duty_gj_ton(T_calciner_degC, plr, operating_hours)
        E_el     = self.electrical_kwh_ton(plr, operating_hours)
        E_liq    = self.liquefaction_energy_gj_ton(injection_pressure_bar)
        E_total  = Q_calc + E_el / 277.78 + E_liq
        cap_pct  = self.solvent_capacity_pct(operating_hours)

        return {
            "capture_rate_kgh":           co2_kgh,
            "calciner_duty_gj_ton":       Q_calc,
            "electrical_kwh_ton":         E_el,
            "liquefaction_energy_gj_ton": E_liq,
            "total_specific_energy_gj_ton": E_total,
            "solvent_capacity_pct":       cap_pct,
        }
