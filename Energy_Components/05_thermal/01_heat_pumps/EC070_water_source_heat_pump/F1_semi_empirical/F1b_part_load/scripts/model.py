"""
EC070 — Water-Source Heat Pump — F1b Part-Load Model

Extends F1a (Carnot-fraction COP) with three additional effects:

1. Part-load factor (EN 14825 with WSHP-calibrated C_d=0.15):
       PLF = 1 - C_d * (1 - PLR)
       COP_PL = COP_full * PLF

2. Cycling losses below PLR_min=0.30 (on/off regime):
       cycling_penalty = 1 - cycling_loss_factor * (PLR_min - PLR) / PLR_min
   WSHPs have lower C_d than ASHPs (more stable source temperature means
   shorter thermal transients on restart). No defrost penalty applies
   because the water-side source cannot freeze under normal WSHP conditions.

3. Water flow rate effect on evaporator U_A:
       U_A_eff = U_A_rated * (flow_rate / flow_rate_rated)^n
   with n=0.8 (turbulent Dittus-Boelter exponent).

   Lower flow reduces evaporator heat transfer, increasing the refrigerant
   evaporation temperature lift => effective T_source_eff < T_source_water.
   We model this as a penalty on the effective source temperature:
       delta_T_UA = Q_evap / U_A_eff - Q_evap / U_A_rated
       T_source_eff = T_source - delta_T_UA

   This captures the flow-rate sensitivity at F1 fidelity without a full
   cycle simulation.

References:
    EN 14825:2016 — Seasonal performance of heat pumps.
    ASHRAE Handbook HVAC Systems and Equipment (2020), Ch. 9.
    Bagarella et al. (2016). Appl. Thermal Eng. 103, 10-21.
    Dittus & Boelter (1930); Gnielinski (1976). (U_A flow exponent)
"""

import numpy as np


class WaterSourceHPF1b:
    """Water-source heat pump with part-load COP + water flow rate effect."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.rated_capacity = u["rated_capacity"]["value"]        # kW_th
        self.carnot_fraction = u["carnot_fraction"]["value"]
        self.aux_power = u["auxiliary_power"]["value"]             # kW
        self.C_d = u["C_d"]["value"]                              # PLF degradation coeff
        self.PLR_min = u["PLR_min"]["value"]                      # cycling threshold
        self.cycling_loss = u["cycling_loss_factor"]["value"]
        self.startup_penalty = u["startup_penalty_kw"]["value"]   # kW
        self.UA_rated = u["UA_rated"]["value"]                    # kW/K
        self.flow_rated = u["flow_rate_rated"]["value"]           # L/s
        self.n_flow = u["flow_rate_exponent"]["value"]

    # ------------------------------------------------------------------
    # Full-load COP (same as F1a)
    # ------------------------------------------------------------------

    def cop_full_load(self, T_source_c, T_sink_c):
        """Full-load heating COP from Carnot fraction approach."""
        T_source = np.asarray(T_source_c, dtype=float) + 273.15
        T_sink = np.asarray(T_sink_c, dtype=float) + 273.15
        dT = T_sink - T_source
        cop_carnot = np.where(dT > 0, T_sink / dT, 30.0)
        return np.clip(self.carnot_fraction * cop_carnot, 1.0, 20.0)

    # ------------------------------------------------------------------
    # Part-load factor (EN 14825)
    # ------------------------------------------------------------------

    def part_load_factor(self, plr):
        """PLF = 1 - C_d * (1 - PLR). At PLR=1: PLF=1; at PLR=0: PLF=1-C_d."""
        plr = np.asarray(plr, dtype=float)
        plf = 1.0 - self.C_d * (1.0 - plr)
        return np.clip(plf, 0.1, 1.0)

    # ------------------------------------------------------------------
    # Water flow rate effect on effective source temperature
    # ------------------------------------------------------------------

    def ua_effective(self, water_flow_ls):
        """U_A_eff = U_A_rated * (flow / flow_rated)^n."""
        q = np.asarray(water_flow_ls, dtype=float)
        q = np.maximum(q, 0.01)  # avoid division by zero
        return self.UA_rated * (q / self.flow_rated) ** self.n_flow

    def effective_source_temperature(self, T_source_c, water_flow_ls, plr=1.0):
        """
        Effective refrigerant-side source temperature accounting for evaporator
        heat transfer resistance at off-rated flow.

            Q_evap = rated_capacity * PLR  (approx.)
            delta_T = Q_evap * (1/U_A_eff - 1/U_A_rated)
            T_source_eff = T_source - delta_T

        At rated flow: delta_T=0, T_source_eff = T_source.
        At reduced flow: U_A_eff < U_A_rated => more temperature approach
        needed => T_source_eff drops => COP drops.
        """
        plr = np.asarray(plr, dtype=float)
        q_evap = self.rated_capacity * np.clip(plr, 0.0, 1.0)  # approx evaporator load
        ua_eff = self.ua_effective(water_flow_ls)
        delta_T = q_evap * (1.0 / ua_eff - 1.0 / self.UA_rated)
        return np.asarray(T_source_c, dtype=float) - delta_T

    # ------------------------------------------------------------------
    # COP with all part-load effects
    # ------------------------------------------------------------------

    def cop(self, T_source_c, T_sink_c, plr=1.0, water_flow_ls=None):
        """
        Part-load COP combining:
          (a) PLF degradation (EN 14825),
          (b) cycling losses below PLR_min,
          (c) reduced-flow evaporator penalty.

        Note: no defrost penalty — water source does not freeze.
        """
        plr = np.asarray(plr, dtype=float)

        if water_flow_ls is None:
            water_flow_ls = self.flow_rated

        T_src_eff = self.effective_source_temperature(T_source_c, water_flow_ls, plr)
        cop_fl = self.cop_full_load(T_src_eff, T_sink_c)
        plf = self.part_load_factor(plr)

        cop_pl = cop_fl * plf

        # Cycling penalty below PLR_min
        cycling_penalty = np.where(
            plr < self.PLR_min,
            1.0 - self.cycling_loss * (self.PLR_min - plr) / self.PLR_min,
            1.0,
        )
        cop_pl = cop_pl * cycling_penalty

        return np.clip(cop_pl, 1.0, 20.0)

    def cop_degradation_factor(self, plr, T_source_c=15.0,
                               T_sink_c=45.0, water_flow_ls=None):
        """
        Overall COP degradation factor = COP_PL / COP_full.
        Combines PLF, cycling penalty, and flow penalty.
        """
        if water_flow_ls is None:
            water_flow_ls = self.flow_rated

        plr = np.asarray(plr, dtype=float)
        cop_full = self.cop_full_load(T_source_c, T_sink_c)
        cop_pl = self.cop(T_source_c, T_sink_c, plr, water_flow_ls)
        return np.where(cop_full > 0, cop_pl / cop_full, 0.0)

    # ------------------------------------------------------------------
    # Capacity and electrical input
    # ------------------------------------------------------------------

    def heating_capacity(self, T_source_c, T_sink_c, plr=1.0):
        """Heating output = rated_capacity * PLR."""
        return self.rated_capacity * np.clip(np.asarray(plr, dtype=float), 0.0, 1.0)

    def electrical_input(self, T_source_c, T_sink_c, plr=1.0, water_flow_ls=None):
        """
        Electrical input including aux power and startup penalty during cycling.
        W = Q / COP + W_aux  (+startup if PLR < PLR_min)
        """
        if water_flow_ls is None:
            water_flow_ls = self.flow_rated

        plr = np.asarray(plr, dtype=float)
        q = self.heating_capacity(T_source_c, T_sink_c, plr)
        c = self.cop(T_source_c, T_sink_c, plr, water_flow_ls)

        w = q / c + self.aux_power
        startup_extra = np.where(plr < self.PLR_min, self.startup_penalty, 0.0)
        return w + startup_extra

    def cooling_cop(self, T_source_c, T_sink_c, plr=1.0, water_flow_ls=None):
        """COP in cooling mode: EER = COP_heat - 1."""
        cop_h = self.cop(T_source_c, T_sink_c, plr, water_flow_ls)
        return np.clip(cop_h - 1.0, 0.5, 19.0)
