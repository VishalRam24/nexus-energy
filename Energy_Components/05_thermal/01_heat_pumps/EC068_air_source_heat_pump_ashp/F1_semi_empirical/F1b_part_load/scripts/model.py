"""
EC068 — Air-Source Heat Pump — F1b Part-Load Model

Extends F1a (Carnot-fraction COP) with:
  1. Part-load factor:  PLF = 1 - C_d * (1 - PLR)        [EN 14825]
     COP_PL = COP_full * PLF
  2. Cycling losses at PLR < PLR_min (on/off regime):
     Additional startup penalty and transient losses.

The degradation coefficient C_d accounts for compressor cycling
inefficiency. EN 14825 default is C_d = 0.25.

References:
    EN 14825:2016 — Seasonal performance of heat pumps.
    AHRI Standard 210/240 — Performance rating of heat pumps.
    Staffell et al. (2012). Energy Environ. Sci., 5, 9291-9306.
"""

import numpy as np


class ASHPF1b:
    """Air-source heat pump with part-load COP degradation (EN 14825)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.rated_capacity = u["rated_capacity"]["value"]      # kW_th
        self.carnot_fraction = u["carnot_fraction"]["value"]
        self.aux_power = u["auxiliary_power"]["value"]           # kW
        self.C_d = u["C_d"]["value"]                            # degradation coeff
        self.PLR_min = u["PLR_min"]["value"]                    # min modulation
        self.cycling_loss = u["cycling_loss_factor"]["value"]   # fraction
        self.startup_penalty = u["startup_penalty_kw"]["value"] # kW

    # ------------------------------------------------------------------
    # Full-load COP (same as F1a)
    # ------------------------------------------------------------------

    def cop_full_load(self, T_source_c, T_sink_c):
        """Full-load COP from Carnot fraction approach."""
        T_source = np.asarray(T_source_c, dtype=float) + 273.15
        T_sink = np.asarray(T_sink_c, dtype=float) + 273.15
        dT = T_sink - T_source
        cop_carnot = np.where(dT > 0, T_sink / dT, 20.0)
        cop = self.carnot_fraction * cop_carnot
        return np.clip(cop, 1.0, 15.0)

    # ------------------------------------------------------------------
    # Part-load factor (EN 14825)
    # ------------------------------------------------------------------

    def part_load_factor(self, plr):
        """
        PLF = 1 - C_d * (1 - PLR)

        At PLR=1: PLF=1 (no degradation).
        At PLR=0: PLF=1-C_d (maximum degradation).
        """
        plr = np.asarray(plr, dtype=float)
        plf = 1.0 - self.C_d * (1.0 - plr)
        return np.clip(plf, 0.1, 1.0)

    # ------------------------------------------------------------------
    # COP with part-load degradation
    # ------------------------------------------------------------------

    def cop(self, T_source_c, T_sink_c, plr=1.0):
        """
        Part-load COP:
            COP_PL = COP_full * PLF(PLR)

        Below PLR_min, additional cycling losses apply.
        """
        plr = np.asarray(plr, dtype=float)
        cop_fl = self.cop_full_load(T_source_c, T_sink_c)
        plf = self.part_load_factor(plr)

        cop_pl = cop_fl * plf

        # Below PLR_min: on/off cycling adds further penalty
        cycling_penalty = np.where(
            plr < self.PLR_min,
            1.0 - self.cycling_loss * (self.PLR_min - plr) / self.PLR_min,
            1.0,
        )
        cop_pl = cop_pl * cycling_penalty

        return np.clip(cop_pl, 1.0, 15.0)

    def cop_degradation_factor(self, plr):
        """
        Overall COP degradation factor = COP_PL / COP_full.
        Combines PLF and cycling penalty.
        """
        plr = np.asarray(plr, dtype=float)
        plf = self.part_load_factor(plr)
        cycling_penalty = np.where(
            plr < self.PLR_min,
            1.0 - self.cycling_loss * (self.PLR_min - plr) / self.PLR_min,
            1.0,
        )
        return plf * cycling_penalty

    # ------------------------------------------------------------------
    # Capacity and electrical input
    # ------------------------------------------------------------------

    def heating_capacity(self, T_source_c, T_sink_c, plr=1.0):
        """Heating output in kW at given part-load ratio."""
        return self.rated_capacity * np.asarray(plr, dtype=float)

    def electrical_input(self, T_source_c, T_sink_c, plr=1.0):
        """
        Compressor + auxiliary electrical input in kW.

        Below PLR_min, startup penalty adds to electrical consumption.
        """
        plr = np.asarray(plr, dtype=float)
        q = self.heating_capacity(T_source_c, T_sink_c, plr)
        c = self.cop(T_source_c, T_sink_c, plr)

        w_comp = q / c + self.aux_power

        # Startup penalty during cycling
        startup_extra = np.where(plr < self.PLR_min, self.startup_penalty, 0.0)

        return w_comp + startup_extra
