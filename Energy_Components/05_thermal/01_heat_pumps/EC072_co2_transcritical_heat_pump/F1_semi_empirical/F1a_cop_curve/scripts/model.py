"""
EC072 — CO2 Transcritical Heat Pump — F1a COP Curve Model

Transcritical R744 cycle: the high side is supercritical, so a "gas cooler"
replaces the condenser and the heat-rejection temperature glides between
T_gc_in ≈ T_compressor_discharge and T_gc_out ≈ T_water_in + pinch.
This makes the COP particularly sensitive to T_water_out (water outlet
temperature on the heated stream) and to the high-side pressure P_high,
which has a clear optimum (Liao et al., 2000).

Approach (lightweight semi-empirical):

    T_sink_eff   = (T_water_in + T_water_out) / 2 + pinch    [K]
    COP_carnot   = T_sink_eff / (T_sink_eff - T_evap)
    f_pressure   = exp( -alpha * (T_water_out - T_water_out_design)**2 )
    COP          = eta_base * f_pressure * COP_carnot

The Liao-style optimum-pressure correlation is exposed for reference:

    P_high_opt   = a * T_gc_out + b * T_gc_out**2 + c

Reference:
    Lorentzen, G. (1994). Revival of carbon dioxide as a refrigerant.
        Int. J. Refrigeration 17(5), 292-301.
    Sarkar, J., Bhattacharyya, S., Ram Gopal, M. (2004). Optimization of a
        transcritical CO2 heat pump cycle for simultaneous cooling and heating.
        Int. J. Refrigeration 27, 830-838.
    Liao, S.M., Zhao, T.S., Jakobsen, A. (2000). Appl. Thermal Eng. 20, 831-841.
"""

import numpy as np


class CO2TranscriticalHPF1a:
    """Transcritical CO2 heat pump — log-mean sink, optimum-pressure penalty COP."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.rated_capacity = u["rated_heating_capacity"]["value"]   # kW_th
        self.eta_base       = u["carnot_fraction_base"]["value"]
        self.pinch          = u["T_gc_out_pinch"]["value"]           # K
        self.aux_power      = u["auxiliary_power"]["value"]          # kW_e
        self.a = u["P_high_opt_a"]["value"]
        self.b = u["P_high_opt_b"]["value"]
        self.c = u["P_high_opt_c"]["value"]
        # Penalty curvature: COP drops ~10% when T_water_out shifts 25 K
        # from the design point (transcritical CO2 favours large dT water).
        self._t_water_out_design = 65.0   # degC
        self._alpha = 4.0e-4              # 1/K2

    # ------------------------------------------------------------------
    def _T_sink_effective_K(self, T_water_in_c, T_water_out_c):
        T_in  = np.asarray(T_water_in_c,  dtype=float)
        T_out = np.asarray(T_water_out_c, dtype=float)
        return 0.5 * (T_in + T_out) + self.pinch + 273.15

    def cop(self, T_evap_c, T_water_in_c, T_water_out_c):
        """Heating COP for the transcritical CO2 cycle."""
        T_evap = np.asarray(T_evap_c, dtype=float) + 273.15
        T_sink = self._T_sink_effective_K(T_water_in_c, T_water_out_c)
        dT = T_sink - T_evap
        cop_carnot = np.where(dT > 0, T_sink / dT, 20.0)
        T_out = np.asarray(T_water_out_c, dtype=float)
        f_press = np.exp(-self._alpha * (T_out - self._t_water_out_design) ** 2)
        cop = self.eta_base * f_press * cop_carnot
        return np.clip(cop, 0.8, 8.0)

    def heating_capacity(self, T_evap_c, T_water_in_c, T_water_out_c, plr=1.0):
        return self.rated_capacity * np.asarray(plr, dtype=float)

    def electrical_input(self, T_evap_c, T_water_in_c, T_water_out_c, plr=1.0):
        q = self.heating_capacity(T_evap_c, T_water_in_c, T_water_out_c, plr)
        c = self.cop(T_evap_c, T_water_in_c, T_water_out_c)
        return q / c + self.aux_power

    def optimum_high_pressure(self, T_gc_out_c):
        """Liao-style optimum gas-cooler-outlet pressure [bar]."""
        T = np.asarray(T_gc_out_c, dtype=float)
        return self.a * T + self.b * T ** 2 + self.c
