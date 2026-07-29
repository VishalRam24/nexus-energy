"""
EC156 — Geothermal Heat Pump (GHP) — F1a COP Map Model

Ground-coupled heat pump (closed-loop GSHP) exploiting stable ground temperature.
Superior to ASHP because ground temperature (10–15°C year-round) is:
  - Higher than ambient air in winter → better heating COP
  - Lower than ambient air in summer → better cooling COP

COP (heating) = eta_Carnot_fraction * T_sink / (T_sink - T_source)  [Carnot fraction approach]
COP (cooling) = COP_heating - 1                                       [thermodynamic identity]
Q_heating     = COP_heating * W_compressor
Q_cooling     = COP_cooling * W_compressor

Compared to ASHP (EC068):
  - Higher carnot_fraction (0.55 vs 0.45) due to more stable source temperature
  - Narrower T_source range (0–25°C vs -20–35°C for ASHP)
  - COP advantage at winter design point is 1–2 COP units over ASHP

Reference:
    Staffell, I. et al. (2012). Energy Environ. Sci., 5, 9291-9306.
    ASHRAE (2011). Geothermal Heating and Cooling Design Guide.
    Lund, J.W. (2010). Geothermics, 39(2), 185-193.
"""

import numpy as np


class GHPF1a:
    """
    Geothermal Heat Pump (GHP) — COP as a function of ground loop and load temperatures.
    Ground source provides stable T_source, improving COP vs ASHP.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.rated_capacity  = u["rated_capacity"]["value"]    # kW_th
        self.carnot_fraction = u["carnot_fraction"]["value"]   # dimensionless
        self.aux_power       = u["auxiliary_power"]["value"]   # kW

    def cop_heating(self, T_source_c, T_sink_c):
        """
        Heating COP from Carnot fraction approach.
        COP_h = eta_Carnot * T_sink / (T_sink - T_source)

        Parameters
        ----------
        T_source_c : float or array — ground loop fluid temperature (degC), typically 5–20°C
        T_sink_c   : float or array — heating load supply temperature (degC), typically 30–55°C

        Returns
        -------
        COP_heating : float or array — heating coefficient of performance
        """
        T_source = np.asarray(T_source_c, dtype=float) + 273.15
        T_sink   = np.asarray(T_sink_c,   dtype=float) + 273.15
        dT = T_sink - T_source
        cop_carnot = np.where(dT > 0, T_sink / dT, 30.0)  # cap at high COP for near-zero dT
        cop = self.carnot_fraction * cop_carnot
        return np.clip(cop, 1.0, 20.0)

    def cop_cooling(self, T_source_c, T_sink_c):
        """
        Cooling COP using Carnot fraction approach for reversed cycle.
        In cooling mode: ground loop rejects heat (hot side = T_source from heating perspective),
        conditioned space is cooled (cold side = T_sink from heating perspective, but lower T).

        COP_c = eta_Carnot_frac * T_cold / (T_hot - T_cold)
        where T_cold = space/chilled-water temperature (T_sink in heating view)
              T_hot  = ground rejection temperature (T_source in heating view, but now as heat sink)

        Parameters
        ----------
        T_source_c : float or array — ground loop temperature (degC) — heat rejection side in cooling
        T_sink_c   : float or array — conditioned space / chilled water temperature (degC)

        Returns
        -------
        COP_cooling : float or array — cooling COP (EER/3.412 in SI)
        """
        T_cold = np.asarray(T_sink_c,   dtype=float) + 273.15   # space to be cooled
        T_hot  = np.asarray(T_source_c, dtype=float) + 273.15   # ground (heat rejection)
        dT = T_hot - T_cold
        cop_carnot_c = np.where(dT > 0, T_cold / dT, 30.0)
        cop_c = self.carnot_fraction * cop_carnot_c
        return np.clip(cop_c, 0.5, 20.0)

    def heating_capacity(self, T_source_c, T_sink_c, plr=1.0):
        """Heating thermal output at part-load ratio (kW_th)."""
        return self.rated_capacity * np.asarray(plr, dtype=float)

    def electrical_input_heating(self, T_source_c, T_sink_c, plr=1.0):
        """Compressor + auxiliary electrical input in heating mode (kW)."""
        q   = self.heating_capacity(T_source_c, T_sink_c, plr)
        cop = self.cop_heating(T_source_c, T_sink_c)
        return q / cop + self.aux_power

    def electrical_input_cooling(self, T_source_c, T_load_c, plr=1.0):
        """Compressor + auxiliary electrical input in cooling mode (kW)."""
        q   = self.rated_capacity * np.asarray(plr, dtype=float)
        cop = self.cop_cooling(T_source_c, T_load_c)
        return q / cop + self.aux_power

    def cop_advantage_over_ashp(self, T_source_ghp_c, T_source_ashp_c, T_sink_c,
                                  ashp_carnot_fraction=0.45):
        """
        Delta-COP advantage of GHP over ASHP at same load conditions.
        Illustrates the benefit of stable ground temperature vs variable ambient.

        Parameters
        ----------
        T_source_ghp_c  : GHP ground loop temperature (degC)
        T_source_ashp_c : ASHP ambient air temperature (degC) — variable, typically lower in winter
        T_sink_c        : Load supply temperature (degC)
        ashp_carnot_fraction : ASHP Carnot fraction (default 0.45, lower than GHP)

        Returns
        -------
        delta_cop : COP_GHP - COP_ASHP (should be positive when ground warmer than air)
        """
        T_src_ghp  = np.asarray(T_source_ghp_c,  dtype=float) + 273.15
        T_src_ashp = np.asarray(T_source_ashp_c, dtype=float) + 273.15
        T_sink     = np.asarray(T_sink_c,         dtype=float) + 273.15
        dT_ghp  = T_sink - T_src_ghp
        dT_ashp = T_sink - T_src_ashp
        cop_ghp  = np.clip(self.carnot_fraction   * np.where(dT_ghp  > 0, T_sink / dT_ghp,  30.0), 1.0, 20.0)
        cop_ashp = np.clip(ashp_carnot_fraction   * np.where(dT_ashp > 0, T_sink / dT_ashp, 30.0), 1.0, 20.0)
        return cop_ghp - cop_ashp
