"""
EC072 — CO2 Transcritical Heat Pump — F1b Part-Load Model

Extends F1a (optimum-P_high COP curve) with:

  1. T_water_in effect on optimum gas-cooler pressure and COP.
     Higher T_water_in → smaller temperature glide in the gas cooler →
     smaller heat extraction for the same P_high → reduced COP.

       f_T_w_in = exp(-gamma * max(T_water_in - T_w_in_design, 0))

     Penalty applies only when T_water_in > design, as warming the inlet
     degrades gas-cooler effectiveness (Sarkar et al., 2004, Fig.4).

  2. Part-load COP degradation via EN-14825-style PLF:
       PLF = 1 - C_d * (1 - PLR)
       COP_pl = COP_full * PLF

  3. On/off cycling losses below PLR_min.

References:
    Lorentzen, G. (1994). Int. J. Refrigeration 17, 292-301.
    Sarkar, J., Bhattacharyya, S., Ram Gopal, M. (2004). Int. J. Refrigeration 27, 830-838.
    Liao, S.M., Zhao, T.S., Jakobsen, A. (2000). Appl. Thermal Eng. 20, 831-841.
    Kim, M.H., Pettersen, J., Bullard, C.W. (2004). Progress in Energy and Combustion Sci. 30, 119-174.
"""

import numpy as np


class CO2TranscriticalHPF1b:
    """Transcritical CO2 HP with part-load, T_water_in, and cycling effects."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.rated_capacity   = u["rated_heating_capacity"]["value"]
        self.eta_base         = u["carnot_fraction_base"]["value"]
        self.pinch            = u["T_gc_out_pinch"]["value"]
        self.aux_power        = u["auxiliary_power"]["value"]
        self.a = u["P_high_opt_a"]["value"]
        self.b = u["P_high_opt_b"]["value"]
        self.c = u["P_high_opt_c"]["value"]
        self.C_d              = u["C_d"]["value"]
        self.PLR_min          = u["PLR_min"]["value"]
        self.cycling_loss     = u["cycling_loss_factor"]["value"]
        self.startup_penalty  = u["startup_penalty_kw"]["value"]
        self.gamma_T_w_in     = u["gamma_T_w_in"]["value"]
        self.T_w_in_design    = u["T_w_in_design"]["value"]
        self._t_water_out_design = 65.0
        self._alpha = 4.0e-4

    # ------------------------------------------------------------------
    # F1a core COP (same equations)
    # ------------------------------------------------------------------

    def _T_sink_eff_K(self, T_water_in_c, T_water_out_c):
        T_in  = np.asarray(T_water_in_c,  dtype=float)
        T_out = np.asarray(T_water_out_c, dtype=float)
        return 0.5 * (T_in + T_out) + self.pinch + 273.15

    def _cop_full_load(self, T_evap_c, T_water_in_c, T_water_out_c):
        """Full-load COP from F1a model."""
        T_evap = np.asarray(T_evap_c, dtype=float) + 273.15
        T_sink = self._T_sink_eff_K(T_water_in_c, T_water_out_c)
        dT = T_sink - T_evap
        cop_carnot = np.where(dT > 0, T_sink / dT, 20.0)
        T_out = np.asarray(T_water_out_c, dtype=float)
        f_press = np.exp(-self._alpha * (T_out - self._t_water_out_design) ** 2)
        return np.clip(self.eta_base * f_press * cop_carnot, 0.8, 8.0)

    # ------------------------------------------------------------------
    # T_water_in correction
    # ------------------------------------------------------------------

    def f_T_water_in(self, T_water_in_c):
        """
        COP penalty when T_water_in > T_w_in_design.

        f = exp(-gamma * max(T_w_in - T_design, 0))

        Higher inlet T shrinks the gas-cooler temperature driving force,
        reducing heat extraction effectiveness.
        """
        T_wi = np.asarray(T_water_in_c, dtype=float)
        excess = np.maximum(T_wi - self.T_w_in_design, 0.0)
        return np.exp(-self.gamma_T_w_in * excess)

    # ------------------------------------------------------------------
    # Part-load factor
    # ------------------------------------------------------------------

    def part_load_factor(self, plr):
        plr = np.asarray(plr, dtype=float)
        plf = 1.0 - self.C_d * (1.0 - plr)
        return np.clip(plf, 0.1, 1.0)

    # ------------------------------------------------------------------
    # Full corrected COP
    # ------------------------------------------------------------------

    def cop(self, T_evap_c, T_water_in_c, T_water_out_c, plr=1.0):
        """
        COP = COP_F1a * f_T_water_in * PLF * f_cycling
        """
        plr = np.asarray(plr, dtype=float)
        cop_fl = self._cop_full_load(T_evap_c, T_water_in_c, T_water_out_c)
        f_twin = self.f_T_water_in(T_water_in_c)
        plf    = self.part_load_factor(plr)

        cycling_penalty = np.where(
            plr < self.PLR_min,
            1.0 - self.cycling_loss * (self.PLR_min - plr) / self.PLR_min,
            1.0,
        )

        cop_pl = cop_fl * f_twin * plf * cycling_penalty
        return np.clip(cop_pl, 0.8, 8.0)

    def cop_degradation_factor(self, T_water_in_c, plr):
        """Overall degradation factor: f_T_water_in * PLF * f_cycling."""
        plr = np.asarray(plr, dtype=float)
        f_twin = self.f_T_water_in(T_water_in_c)
        plf    = self.part_load_factor(plr)
        cycling_penalty = np.where(
            plr < self.PLR_min,
            1.0 - self.cycling_loss * (self.PLR_min - plr) / self.PLR_min,
            1.0,
        )
        return f_twin * plf * cycling_penalty

    # ------------------------------------------------------------------
    # Capacity and power
    # ------------------------------------------------------------------

    def heating_capacity(self, plr=1.0):
        return self.rated_capacity * np.asarray(plr, dtype=float)

    def electrical_input(self, T_evap_c, T_water_in_c, T_water_out_c, plr=1.0):
        plr = np.asarray(plr, dtype=float)
        q = self.heating_capacity(plr)
        c = self.cop(T_evap_c, T_water_in_c, T_water_out_c, plr)
        w_comp = q / np.where(c > 1e-6, c, 1e-6) + self.aux_power
        startup_extra = np.where(plr < self.PLR_min, self.startup_penalty, 0.0)
        return w_comp + startup_extra

    def optimum_high_pressure(self, T_gc_out_c):
        """Liao-style optimum gas-cooler-outlet pressure [bar]."""
        T = np.asarray(T_gc_out_c, dtype=float)
        return self.a * T + self.b * T ** 2 + self.c
