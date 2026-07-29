"""
EC140 — Anaerobic Digester (Mesophilic) — F1a Biogas Yield Model

Equations:
    methane_yield = Y_max * (1 - exp(-k * HRT))                      [m³_CH4/kgVS]
    temperature correction (Arrhenius for T <= T_inhibit):
        f_T = exp(-E_a/R * (1/T - 1/T_ref))
    above T_inhibit (42 degC), yield drops linearly to 0 at 55 degC:
        f_T = f_T(T_inhibit) * (55 - T) / (55 - T_inhibit)

    Y_corrected      = methane_yield * f_T                            [m³_CH4/kgVS]
    VS_mass_per_day  = vs_loading * V_reactor                         [kgVS/day]
    methane_rate     = Y_corrected * VS_mass_per_day                  [m³_CH4/day]
    biogas_rate      = methane_rate / methane_fraction                [m³_biogas/day]
    energy_output    = methane_rate * LHV_methane                     [kWh/day]

References:
    Buswell, A.M. & Mueller, H.F. (1952). Mechanism of methane fermentation.
    Ind. Eng. Chem., 44(3), 550-552.
    Batstone, D.J. et al. (2002). Anaerobic Digestion Model No.1 (ADM1).
    IWA Publishing. (simplified yield sub-model)
"""

import numpy as np


class AnaerobicDigesterF1a:
    """Mesophilic anaerobic digester — simplified yield model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V = u["V_reactor_m3"]["value"]              # m³
        self.Y_max = u["Y_max_m3kgvs"]["value"]          # m³_CH4/kgVS
        self.k = u["k_decay_per_day"]["value"]            # 1/day
        self.T_ref = u["T_ref_k"]["value"]               # K
        self.E_a = u["E_a_jmol"]["value"]                # J/mol
        self.R = u["R_jmolk"]["value"]                   # J/(mol·K)
        self.f_ch4 = u["methane_fraction"]["value"]      # -
        self.LHV = u["LHV_methane_kwh_m3"]["value"]      # kWh/m³_CH4
        self.T_inhibit_k = u["T_inhibit_c"]["value"] + 273.15   # K
        self.T_max_k = 55.0 + 273.15                    # K (upper kill temperature)

    def temperature_factor(self, temp_c):
        """
        Temperature correction factor f_T [-].
        Arrhenius below T_inhibit (42 degC), linear decay above to 0 at 55 degC.
        Normalized so f_T(T_ref=37 degC) = 1.0.
        """
        T = np.asarray(temp_c, dtype=float) + 273.15  # K
        # Arrhenius factor
        f_arr = np.exp(-self.E_a / self.R * (1.0 / T - 1.0 / self.T_ref))
        # f_T at inhibition threshold (for normalization of the linear decline)
        f_inhibit = float(np.exp(-self.E_a / self.R * (1.0 / self.T_inhibit_k - 1.0 / self.T_ref)))
        # Linear decline above T_inhibit
        f_linear = f_inhibit * (self.T_max_k - T) / (self.T_max_k - self.T_inhibit_k)
        f_linear = np.clip(f_linear, 0.0, f_inhibit)
        # Apply: Arrhenius below inhibit, linear above
        f_T = np.where(T <= self.T_inhibit_k, f_arr, f_linear)
        return np.clip(f_T, 0.0, None)

    def methane_yield(self, hrt, temp_c=37.0):
        """
        Specific methane yield [m³_CH4/kgVS].
        Y = Y_max * (1 - exp(-k * HRT)) * f_T(temp)
        """
        HRT = np.asarray(hrt, dtype=float)
        f_T = self.temperature_factor(temp_c)
        Y_base = self.Y_max * (1.0 - np.exp(-self.k * HRT))
        return Y_base * f_T

    def biogas_rate(self, vs_loading, hrt, temp_c=37.0):
        """
        Total biogas production rate [m³_biogas/day].
        biogas_rate = methane_rate / methane_fraction
        """
        return self.methane_rate(vs_loading, hrt, temp_c) / self.f_ch4

    def methane_rate(self, vs_loading, hrt, temp_c=37.0):
        """
        Methane production rate [m³_CH4/day].
        """
        vs = np.asarray(vs_loading, dtype=float)
        Y = self.methane_yield(hrt, temp_c)
        VS_day = vs * self.V   # kgVS/day
        return Y * VS_day

    def energy_output(self, vs_loading, hrt, temp_c=37.0):
        """
        Equivalent thermal energy from methane [kWh/day].
        energy = methane_rate * LHV_methane
        """
        return self.methane_rate(vs_loading, hrt, temp_c) * self.LHV
