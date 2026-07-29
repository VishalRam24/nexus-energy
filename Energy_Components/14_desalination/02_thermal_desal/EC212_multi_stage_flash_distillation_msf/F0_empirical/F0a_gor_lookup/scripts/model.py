"""EC212 MSF F0a — empirical GOR-vs-top-brine-temperature lookup.

MSF Gain Output Ratio (kg distillate / kg steam) increases with top brine
temperature as more flashing range becomes available. 1-D np.interp over a
tabulated (T_top, GOR) breakpoint array; thermal SEC is the rated datasheet
value (latent heat / GOR cross-check available).

Source: El-Dessouky & Ettouney (2002); IDA (reused from EC212 F1a). NumPy only.
"""
import numpy as np


class GORLookup:
    def __init__(self, T_top_bp, gor_bp, T_top_rated, gor_rated,
                 sec_thermal_kJ_kg, sec_elec_kWh_m3, recovery,
                 capacity_m3_h, h_latent_kJ_kg):
        self.T_top_bp = np.asarray(T_top_bp, dtype=float)
        self.gor_bp = np.asarray(gor_bp, dtype=float)
        self.T_top_rated = float(T_top_rated)
        self.gor_rated = float(gor_rated)
        self.sec_thermal_kJ_kg = float(sec_thermal_kJ_kg)
        self.sec_elec_kWh_m3 = float(sec_elec_kWh_m3)
        self.recovery = float(recovery)
        self.capacity_m3_h = float(capacity_m3_h)
        self.h_latent_kJ_kg = float(h_latent_kJ_kg)

    def gor(self, T_top):
        """GOR at given top brine temperature; clamps to endpoints."""
        return np.interp(T_top, self.T_top_bp, self.gor_bp)

    def thermal_sec_from_gor(self, T_top):
        """Thermal SEC (kJ/kg distillate) = latent heat / GOR."""
        return self.h_latent_kJ_kg / self.gor(T_top)

    def distillate_flow(self, load_fraction):
        return self.capacity_m3_h * np.asarray(load_fraction, dtype=float)
