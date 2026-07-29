"""
EC131 — Tidal Barrage — F1b Tidal Range Model

Extends F1a basin model with:
1. Efficiency vs head ratio: eta(h_ratio) = eta_peak * (1 - k_h*(h_ratio-1)^2)
   where h_ratio = h / h_design (tidal amplitude relative to design)
   At low head differentials, turbine efficiency drops.

2. Seawater density correction:
   rho_seawater(T, S) = rho_ref + 0.8 * (S - S_ref) - 0.15 * (T - T_ref)
   where S = salinity [g/kg], T = temperature [°C]

3. Sluicing optimization (pumping vs ebb generation):
   Option: pump water into basin at high tide (adds energy stored in basin head)
   Net energy gain from pumping = rho * g * A * (h_pump^2 - h_ebb^2) * eta_pump * eta_turb
   Simplified: pumping_gain_fraction of ebb energy (given pumping_efficiency parameter)

4. Power curve regions:
   - h < h_min: sluice gates open, no generation
   - h_min <= h < h_design: partial generation (efficiency degrades)
   - h = h_design: peak generation

References:
    Prandle (1984), Phys. Chem. Earth, 9, 217-228.
    Baker (1991), "Tidal Power", Peter Peregrinus.
    Aggidis & Feather (2012), Ocean Engineering, 40, 10-17.
    EMEC (2015), Tidal energy assessment.
"""

import numpy as np


class TidalBarrageF1b:
    """Tidal barrage — efficiency vs head ratio, salinity/temperature density, sluicing."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.A          = u["A_basin"]["value"]              # m2
        self.h_design   = u["h_tidal"]["value"]              # m (tidal amplitude)
        self.T_tide     = u["T_tide"]["value"]               # s
        self.eta_peak   = u["eta_peak"]["value"]
        self.h_min      = u["h_min_operation"]["value"]      # m
        self.rho_ref    = u["rho_ref"]["value"]              # kg/m3
        self.g          = u["g"]["value"]
        self.k_h        = u["k_head_efficiency"]["value"]    # head curvature
        self.S_ref      = u["S_ref"]["value"]                # psu design salinity
        self.T_ref      = u["T_ref"]["value"]                # °C design temperature
        self.pump_frac  = u["pumping_gain_fraction"]["value"] # fraction gain from pumping

    def seawater_density(self, T_C=None, S_psu=None):
        """
        Seawater density [kg/m3] as function of temperature and salinity.

        rho ≈ rho_ref + 0.8*(S - S_ref) - 0.15*(T - T_ref)
        """
        T = self.T_ref if T_C is None else np.asarray(T_C, dtype=float)
        S = self.S_ref if S_psu is None else np.asarray(S_psu, dtype=float)
        return self.rho_ref + 0.8 * (S - self.S_ref) - 0.15 * (T - self.T_ref)

    def turbine_efficiency(self, tidal_amplitude_m):
        """
        Turbine efficiency as function of actual tidal amplitude.

        eta(h) = eta_peak * (1 - k_h*(h/h_design - 1)^2)
        Returns 0 below h_min.
        """
        h = np.asarray(tidal_amplitude_m, dtype=float)
        h_ratio = h / self.h_design
        eta = self.eta_peak * (1.0 - self.k_h * (h_ratio - 1.0) ** 2)
        eta = np.where(h < self.h_min, 0.0, eta)
        return np.clip(eta, 0.0, self.eta_peak)

    def theoretical_avg_power_w(self, tidal_range_m, basin_area_m2=None, T_C=None, S_psu=None):
        """
        Theoretical average power (before turbine/generator losses) [W].

        P_theo = (rho * g * A * h^2) / (2 * T_tide)
        where h = tidal_range / 2.
        """
        R = np.asarray(tidal_range_m, dtype=float)
        h = R / 2.0
        A = self.A if basin_area_m2 is None else float(basin_area_m2)
        rho = self.seawater_density(T_C, S_psu)
        E_cycle = 0.5 * rho * self.g * A * h ** 2
        P_theo = E_cycle / self.T_tide
        P_theo = np.where(h < self.h_min, 0.0, P_theo)
        return np.clip(P_theo, 0.0, None)

    def avg_power_kw(self, tidal_range_m, basin_area_m2=None, T_C=None, S_psu=None,
                     pumping_mode=False):
        """
        Average electrical output [kW] with efficiency vs head and density corrections.

        pumping_mode: if True, adds pumping gain fraction on top of ebb generation.
        """
        h = np.asarray(tidal_range_m, dtype=float) / 2.0
        P_theo_w = self.theoretical_avg_power_w(tidal_range_m, basin_area_m2, T_C, S_psu)
        eta = self.turbine_efficiency(h)

        P_kw = P_theo_w * eta / 1000.0
        if pumping_mode:
            P_kw = P_kw * (1.0 + self.pump_frac)
        return P_kw

    def avg_power_mw(self, tidal_range_m, basin_area_m2=None, T_C=None, S_psu=None,
                     pumping_mode=False):
        return self.avg_power_kw(tidal_range_m, basin_area_m2, T_C, S_psu, pumping_mode) / 1000.0

    def energy_per_cycle_mwh(self, tidal_range_m, basin_area_m2=None, T_C=None, S_psu=None):
        P_mw = self.avg_power_mw(tidal_range_m, basin_area_m2, T_C, S_psu)
        T_h = self.T_tide / 3600.0
        return P_mw * T_h

    def capacity_factor(self, tidal_range_m, P_installed_mw, basin_area_m2=None,
                        T_C=None, S_psu=None):
        P_mw = self.avg_power_mw(tidal_range_m, basin_area_m2, T_C, S_psu)
        return np.clip(P_mw / P_installed_mw, 0.0, 1.0)
