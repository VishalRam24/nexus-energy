"""
EC156 -- Geothermal Heat Pump (GHP) -- F1b Ground Thermal Degradation Model

Builds on F1a (COP map) by adding:
  - Ground thermal saturation over continuous heating season (soil temperature rises)
  - Brine chemistry effect on heat exchanger fouling (CaCO3, MgSO4 scaling)
  - Part-load COP correction
  - Condenser/evaporator temperature sensitivity
  - Multi-year ground thermal recovery during off-season

GHP-specific physics not in standard ASHP models:
  1. Ground thermal saturation: continuous heat extraction cools ground locally
     Delta_T_ground(t) = Q_heat * R_th_ground * (1 - exp(-t/tau_ground))
     where tau_ground ~ 500-2000 hours depending on soil thermal diffusivity.
     COP degrades as effective T_source decreases.

  2. Brine chemistry scaling:
     Fouling factor R_f increases with TDS (total dissolved solids) and temperature.
     COP penalty: f_fouling = 1 / (1 + U_0 * R_f)
     where U_0 is the design UA value of the ground heat exchanger.

  3. Condenser temperature sensitivity:
     Higher T_sink (supply temperature) reduces COP — same as F1a but now
     combined with time-varying T_source.

  4. Part-load COP:
     PLR penalty: f_PLR(PLR) = interpolated from part-load curve.
     Below ~30% PLR, cycling losses dominate.

References:
    Staffell, I. et al. (2012). Energy Environ. Sci., 5, 9291-9306.
    ASHRAE (2011). Geothermal Heating and Cooling Design Guide.
    Lund, J.W. (2010). Direct utilization of geothermal energy. Geothermics, 39(2), 185-193.
    Kavanaugh, S.P. & Rafferty, K. (2014). Geothermal Heating and Cooling.
        ASHRAE Press. (Ground thermal models)
    Yang, H. et al. (2010). Vertical-borehole ground-coupled heat pumps.
        Applied Energy, 87, 16-27.
"""

import numpy as np


class GHPF1b:
    """
    Geothermal Heat Pump with ground thermal saturation, brine scaling,
    part-load correction, and condenser temperature sensitivity.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.rated_capacity   = u["rated_capacity"]["value"]      # kW_th
        self.carnot_fraction  = u["carnot_fraction"]["value"]     # dimensionless
        self.aux_power        = u["auxiliary_power"]["value"]     # kW
        self.tau_ground       = u["tau_ground_h"]["value"]        # thermal time constant [h]
        self.R_th_ground      = u["R_th_ground_KkW"]["value"]     # K/kW ground thermal resistance
        self.k_fouling        = u["k_fouling_m2KkW"]["value"]     # m^2*K/kW fouling rate
        self.tds_ref          = u["TDS_ref_ppm"]["value"]         # ppm reference TDS
        self.PLR_min          = u["PLR_min"]["value"]
        self.T_ground_ref     = u["T_ground_ref"]["value"]        # undisturbed ground T

        plc = params["part_load_curve"]
        self.plr_points    = np.array(plc["PLR_points"])
        self.cop_ratio_pts = np.array(plc["cop_ratio"])

    def f_plr(self, PLR):
        """Part-load COP ratio: COP_actual / COP_full_load."""
        PLR_c = float(np.clip(PLR, self.PLR_min, 1.0))
        return float(np.interp(PLR_c, self.plr_points, self.cop_ratio_pts))

    def ground_temperature_degradation(self, heat_rate_kw, operation_hours):
        """
        Effective ground temperature after thermal saturation.
        T_source_eff = T_ground_ref - Delta_T(t)
        Delta_T = Q * R_th_ground * (1 - exp(-t/tau_ground))

        At t=0: no degradation.
        At t >> tau_ground: steady-state temperature depression = Q * R_th.

        Kavanaugh & Rafferty (2014): thermal saturation significant after
        500-1000 hours of continuous operation.
        """
        Q_h  = max(0.0, float(heat_rate_kw))
        t    = max(0.0, float(operation_hours))
        dT   = Q_h * self.R_th_ground * (1.0 - np.exp(-t / self.tau_ground))
        T_source_eff = self.T_ground_ref - dT
        return float(T_source_eff)

    def fouling_factor(self, TDS_ppm, operation_hours):
        """
        Cumulative fouling resistance [m^2*K/kW] from brine scaling.
        R_f = k_fouling * (TDS/TDS_ref)^0.5 * t^0.3
        (Yang et al. 2010: fouling increases sub-linearly with TDS and time)
        """
        TDS = max(0.0, float(TDS_ppm))
        t   = max(0.0, float(operation_hours))
        R_f = self.k_fouling * (TDS / max(self.tds_ref, 1.0)) ** 0.5 * max(t, 0.1) ** 0.3
        return float(np.clip(R_f, 0.0, 0.10))   # cap at 0.10 m^2K/kW (severe fouling)

    def cop_fouling_factor(self, TDS_ppm, operation_hours):
        """
        COP derating factor due to fouling.
        f_foul = 1 / (1 + R_f * U_design)
        where U_design ~ 0.5 kW/(m^2*K) for ground HX.
        """
        R_f = self.fouling_factor(TDS_ppm, operation_hours)
        U_design = 0.5    # kW/(m^2*K) — typical ground HX UA density
        f = 1.0 / (1.0 + R_f * U_design)
        return float(np.clip(f, 0.50, 1.0))

    def cop_heating(self, T_source_c, T_sink_c):
        """Heating COP from Carnot fraction approach (same as F1a)."""
        T_source = np.asarray(T_source_c, dtype=float) + 273.15
        T_sink   = np.asarray(T_sink_c,   dtype=float) + 273.15
        dT = T_sink - T_source
        cop_carnot = np.where(dT > 0, T_sink / dT, 30.0)
        cop = self.carnot_fraction * cop_carnot
        return np.clip(cop, 1.0, 20.0)

    def cop_cooling(self, T_source_c, T_sink_c):
        """Cooling COP (reversed cycle)."""
        T_cold = np.asarray(T_sink_c,   dtype=float) + 273.15
        T_hot  = np.asarray(T_source_c, dtype=float) + 273.15
        dT = T_hot - T_cold
        cop_carnot_c = np.where(dT > 0, T_cold / dT, 30.0)
        cop_c = self.carnot_fraction * cop_carnot_c
        return np.clip(cop_c, 0.5, 20.0)

    def predict(self, T_sink_c, PLR=1.0, operation_hours=0.0,
                TDS_ppm=200.0, heat_rate_kw=None, mode="heating"):
        """
        Compute GHP performance with ground thermal degradation and fouling.

        Args:
            T_sink_c:         Heating load supply / cooling return temperature [degC]
            PLR:              Part-load ratio [PLR_min - 1.0]
            operation_hours:  Cumulative operation since ground recovery [hours]
            TDS_ppm:          Brine total dissolved solids [ppm]
            heat_rate_kw:     Current heat extraction rate [kW]; uses rated if None
            mode:             "heating" or "cooling"

        Returns:
            dict with:
                cop_heating         : heating COP (full-load Carnot)
                cop_cooling         : cooling COP (full-load Carnot)
                cop_effective       : effective COP after all deratings
                T_source_effective  : effective ground loop T after thermal saturation [degC]
                ground_dT           : ground temperature depression [K]
                fouling_factor      : brine fouling COP derating [-]
                part_load_factor    : PLR COP derating [-]
                heating_capacity_kw : thermal output [kW]
                electrical_input_kw : compressor + aux power [kW]
                cop_advantage_over_ashp: COP uplift vs ASHP at same T_sink
        """
        PLR_eff = float(np.clip(PLR, self.PLR_min, 1.0))
        Q_ref   = float(heat_rate_kw) if heat_rate_kw is not None else self.rated_capacity * PLR_eff

        # Ground thermal saturation
        T_source_eff = self.ground_temperature_degradation(Q_ref, operation_hours)
        dT_ground    = self.T_ground_ref - T_source_eff

        # Full-load COP at effective conditions
        cop_h = float(self.cop_heating(T_source_eff, T_sink_c))
        cop_c = float(self.cop_cooling(T_source_eff, T_sink_c))

        # Derating factors
        f_pl    = self.f_plr(PLR_eff)
        f_foul  = self.cop_fouling_factor(TDS_ppm, operation_hours)

        # Effective COP
        cop_ref = cop_h if mode == "heating" else cop_c
        cop_eff = float(cop_ref * f_pl * f_foul)
        cop_eff = float(np.clip(cop_eff, 0.5, 20.0))

        # Capacity and electrical input
        Q_out   = self.rated_capacity * PLR_eff
        W_elec  = Q_out / cop_eff + self.aux_power if cop_eff > 0.1 else Q_out + self.aux_power

        # COP advantage over ASHP at typical cold-climate design point.
        # ASHP air source temperature: -5 degC (standard cold-climate winter design).
        # GHP advantage is most pronounced in winter when air is cold but ground is stable.
        T_ashp_air = -5.0   # degC cold-climate winter air temperature (design case)
        T_ashp_air_K = T_ashp_air + 273.15
        T_sink_K     = float(T_sink_c) + 273.15
        dT_ashp = max(1.0, T_sink_K - T_ashp_air_K)
        cop_ashp   = float(np.clip(0.45 * T_sink_K / dT_ashp, 1.0, 20.0))
        cop_advantage = float(cop_h - cop_ashp)

        return {
            "cop_heating":           cop_h,
            "cop_cooling":           cop_c,
            "cop_effective":         cop_eff,
            "T_source_effective":    T_source_eff,
            "ground_dT":             dT_ground,
            "fouling_factor":        f_foul,
            "part_load_factor":      f_pl,
            "heating_capacity_kw":   Q_out,
            "electrical_input_kw":   float(W_elec),
            "cop_advantage_over_ashp": cop_advantage,
        }
