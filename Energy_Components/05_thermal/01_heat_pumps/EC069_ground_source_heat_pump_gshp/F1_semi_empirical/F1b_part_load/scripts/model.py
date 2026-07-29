"""
EC069 — Ground-Source Heat Pump (GSHP) — F1b Part-Load Model

Extends F1a (Carnot-fraction COP) with:
  1. Part-load factor: PLF = 1 - C_d * (1 - PLR)    [EN 14825]
  2. Seasonal ground temperature model:
     T_ground(month) = T_mean + A * cos(2*pi*(month - month_min) / 12)

The GSHP has lower C_d than ASHP (0.20 vs 0.25) because the stable
ground source reduces compressor cycling frequency.

References:
    EN 14825:2016 — Seasonal performance of heat pumps.
    ASHRAE Handbook — HVAC Applications (2019), Chapter 34.
    Staffell et al. (2012). Energy Environ. Sci., 5, 9291-9306.
"""

import numpy as np


class GSHPF1b:
    """Ground-source heat pump with part-load + seasonal ground temperature."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.rated_capacity = u["rated_capacity"]["value"]       # kW_th
        self.carnot_fraction = u["carnot_fraction"]["value"]
        self.aux_power = u["auxiliary_power"]["value"]            # kW
        self.C_d = u["C_d"]["value"]
        self.PLR_min = u["PLR_min"]["value"]
        self.T_mean = u["T_ground_mean"]["value"]                # degC
        self.T_amp = u["T_ground_amplitude"]["value"]            # degC
        self.month_min = u["month_min_temp"]["value"]            # month

    # ------------------------------------------------------------------
    # Ground temperature model
    # ------------------------------------------------------------------

    def ground_temperature(self, month):
        """
        Seasonal ground temperature at borehole/loop depth.

        T_ground(month) = T_mean + A * cos(2*pi*(month - month_min)/12)

        At month_min: T = T_mean + A (maximum of cosine => coldest ground
        is at month_min, but the cosine gives max at month_min).
        Convention: month_min is the month of MINIMUM ground temperature,
        so we invert: T = T_mean - A * cos(2*pi*(month - month_min)/12)
        """
        month = np.asarray(month, dtype=float)
        phase = 2.0 * np.pi * (month - self.month_min) / 12.0
        return self.T_mean - self.T_amp * np.cos(phase)

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
        return np.clip(cop, 1.0, 20.0)

    # ------------------------------------------------------------------
    # Part-load factor
    # ------------------------------------------------------------------

    def part_load_factor(self, plr):
        """PLF = 1 - C_d * (1 - PLR)"""
        plr = np.asarray(plr, dtype=float)
        plf = 1.0 - self.C_d * (1.0 - plr)
        return np.clip(plf, 0.1, 1.0)

    # ------------------------------------------------------------------
    # COP with part-load
    # ------------------------------------------------------------------

    def cop(self, T_source_c, T_sink_c, plr=1.0):
        """Part-load COP: COP_full * PLF(PLR)."""
        cop_fl = self.cop_full_load(T_source_c, T_sink_c)
        plf = self.part_load_factor(plr)
        return np.clip(cop_fl * plf, 1.0, 20.0)

    def cop_from_month(self, month, T_sink_c, plr=1.0):
        """COP using seasonal ground temperature model."""
        T_ground = self.ground_temperature(month)
        return self.cop(T_ground, T_sink_c, plr)

    # ------------------------------------------------------------------
    # Capacity and electrical input
    # ------------------------------------------------------------------

    def heating_capacity(self, T_source_c, T_sink_c, plr=1.0):
        """Heating output in kW."""
        return self.rated_capacity * np.asarray(plr, dtype=float)

    def electrical_input(self, T_source_c, T_sink_c, plr=1.0):
        """Compressor + circulating pump electrical input in kW."""
        q = self.heating_capacity(T_source_c, T_sink_c, plr)
        c = self.cop(T_source_c, T_sink_c, plr)
        return q / c + self.aux_power

    def electrical_input_from_month(self, month, T_sink_c, plr=1.0):
        """Electrical input using seasonal ground temperature."""
        T_ground = self.ground_temperature(month)
        return self.electrical_input(T_ground, T_sink_c, plr)
