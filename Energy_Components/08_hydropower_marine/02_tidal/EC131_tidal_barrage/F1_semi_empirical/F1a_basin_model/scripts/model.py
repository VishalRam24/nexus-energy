"""
EC131 — Tidal Barrage — F1a Basin Model

Ebb-generation theoretical power:
    P_theoretical = 0.5 * rho * g * A * h^2 / T_tide   [W]
    P_avg = eta_plant * P_theoretical                   [W]

Where:
    h = tidal_range / 2  (amplitude, half of peak-to-trough)
    A = basin plan area [m2]
    T_tide = tidal period [s]

This is the standard Prandle/Baker energy-per-cycle formula divided by T.
Energy per ebb cycle: E = 0.5 * rho * g * A * h^2
Average power: P = E / T_tide (one ebb per semidiurnal cycle)

Practical efficiency eta = 0.22–0.33 (turbines + generator + civil losses).
No generation below h_min (sluice gates remain closed).

Reference:
    Prandle (1984), "Simple theory of tidal power", Phys. Chem. Earth, 9, 217–228.
    Baker (1991), "Tidal Power", Peter Peregrinus.
    Charlier (2003), "Tidal Energy", Springer.
"""

import numpy as np


class TidalBarrageF1a:
    """Tidal barrage ebb-generation power model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.A = u["A_basin"]["value"]           # m2
        self.h_design = u["h_tidal"]["value"]    # m (amplitude)
        self.T_tide = u["T_tide"]["value"]       # s
        self.eta = u["eta_plant"]["value"]       # dimensionless
        self.rho = u["rho"]["value"]             # kg/m3
        self.g = u["g"]["value"]                 # m/s2
        self.h_min = u["h_min_operation"]["value"]  # m

    def theoretical_avg_power_w(self, tidal_range_m, basin_area_m2=None):
        """
        Theoretical average power (no efficiency losses) in Watts.

        Args:
            tidal_range_m: Full peak-to-trough tidal range [m] (= 2 * amplitude)
            basin_area_m2: Basin area override [m2] (uses A_basin if None)
        """
        R = np.asarray(tidal_range_m, dtype=float)
        h = R / 2.0                              # amplitude
        A = self.A if basin_area_m2 is None else float(basin_area_m2)
        # Energy per ebb cycle: 0.5 * rho * g * A * h^2
        E_cycle = 0.5 * self.rho * self.g * A * h ** 2
        P_theo = E_cycle / self.T_tide           # average power [W]
        # No generation below minimum head
        P_theo = np.where(h < self.h_min, 0.0, P_theo)
        return np.clip(P_theo, 0.0, None)

    def avg_power_kw(self, tidal_range_m, basin_area_m2=None):
        """Average electrical output in kW (with plant efficiency)."""
        P_w = self.theoretical_avg_power_w(tidal_range_m, basin_area_m2)
        return P_w * self.eta / 1000.0

    def avg_power_mw(self, tidal_range_m, basin_area_m2=None):
        """Average electrical output in MW."""
        return self.avg_power_kw(tidal_range_m, basin_area_m2) / 1000.0

    def energy_per_cycle_mwh(self, tidal_range_m, basin_area_m2=None):
        """Electrical energy per tidal cycle in MWh."""
        P_mw = self.avg_power_mw(tidal_range_m, basin_area_m2)
        T_h = self.T_tide / 3600.0
        return P_mw * T_h

    def capacity_factor(self, tidal_range_m, P_installed_mw, basin_area_m2=None):
        """Capacity factor relative to installed capacity."""
        P_mw = self.avg_power_mw(tidal_range_m, basin_area_m2)
        return np.clip(P_mw / P_installed_mw, 0.0, 1.0)
