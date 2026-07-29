"""
EC132 — Tidal Stream Turbine — F1b Density-Depth Model

Extends F1a with:
1. Depth-dependent power correction:
   - Velocity shear profile: v(z) = v_ref * (z / z_ref)^alpha
   - Depth-averaged velocity: v_avg = v_ref * (alpha+1)^(-1) * (H/z_ref)^alpha ... integral
   - For the turbine at hub height z_hub in water depth H:
       v_hub = v_surface * (z_hub / H)^alpha
   - Rotor-swept area captures velocity variation (simplified): same hub velocity

2. Salinity/temperature water density correction:
   rho(T, S) ≈ rho_ref + 0.8*(S - S_ref) - 0.15*(T - T_ref)

3. Blockage effect correction:
   If rotor fills significant fraction of channel cross-section, available kinetic energy
   increases due to flow acceleration around rotor (actuator disk + blockage):
   Cp_effective = Cp * (1 - blockage)^(-n_block)   [simplified]
   Blockage = A_rotor / A_channel

4. Bi-directional operation: ebb and flood have different peak velocities.
   Total daily energy = energy_ebb + energy_flood (using respective peak velocities).

References:
    Fraenkel (2002), Proc. Inst. Mech. Eng. A, 216, 1-14.
    IEC TS 62600-200:2013, Marine energy assessment.
    Garrett & Cummins (2005), Proc. R. Soc. A, 461, 2563-2572 (blockage).
    Bahaj et al. (2007), Renew. Energy, 32, 1587-1597.
"""

import numpy as np

_G = 9.81


class TidalStreamTurbineF1b:
    """Tidal stream turbine — density, depth-shear, blockage, bidirectional."""

    def __init__(self, params: dict):
        t = params["turbine"]
        self.P_rated     = t["rated_power"]["value"]       # kW
        self.D           = t["rotor_diameter"]["value"]    # m
        self.Cp          = t["Cp_rated"]["value"]
        self.v_cut_in    = t["cut_in_speed"]["value"]      # m/s
        self.v_rated     = t["rated_speed"]["value"]       # m/s
        self.v_cut_out   = t["cut_out_speed"]["value"]     # m/s
        self.eta         = t["eta_drivetrain"]["value"]
        self.rho_ref     = t["rho_water"]["value"]         # kg/m3
        self.S_ref       = t["S_ref"]["value"]             # psu
        self.T_ref       = t["T_ref"]["value"]             # °C
        self.alpha       = t["shear_exponent"]["value"]    # velocity shear exponent
        self.z_hub       = t["hub_depth"]["value"]         # m from surface (positive down)
        self.A_channel   = t["channel_cross_section"]["value"]  # m2
        self.n_block     = t["blockage_exponent"]["value"]  # blockage correction power

        self.A = np.pi * (self.D / 2.0) ** 2   # rotor swept area [m2]
        self.blockage = self.A / self.A_channel

    def seawater_density(self, T_C=None, S_psu=None):
        """Seawater density [kg/m3] from temperature and salinity."""
        T = self.T_ref if T_C is None else np.asarray(T_C, dtype=float)
        S = self.S_ref if S_psu is None else np.asarray(S_psu, dtype=float)
        return self.rho_ref + 0.8 * (S - self.S_ref) - 0.15 * (T - self.T_ref)

    def depth_velocity_correction(self, water_depth_m):
        """
        Velocity correction factor for hub depth in the water column.

        v_hub = v_surface * (1 - z_hub/H)^alpha  [z_hub from surface, H = total depth]

        Returns ratio v_hub/v_surface.
        """
        H = np.asarray(water_depth_m, dtype=float)
        H_safe = np.maximum(H, self.z_hub + 0.1)
        # Fraction from surface to hub (0 = surface, 1 = seabed)
        depth_frac = np.clip(1.0 - self.z_hub / H_safe, 0.01, 1.0)
        return depth_frac ** self.alpha

    def blockage_correction(self):
        """
        Cp correction factor for channel blockage.

        Cp_eff = Cp * (1 - blockage)^(-n_block)
        Capped to avoid unphysical values.
        """
        corr = (1.0 - self.blockage) ** (-self.n_block)
        return np.clip(corr, 1.0, 1.25)   # max 25% blockage gain

    def power_kw(self, current_speed_ms, water_density=None, water_depth_m=None,
                 apply_blockage=True):
        """
        Electrical power output [kW].

        Parameters
        ----------
        current_speed_ms : surface tidal current speed [m/s] (scalar or array)
        water_density    : seawater density [kg/m3] (or use T/S via separate call)
        water_depth_m    : water depth [m] for depth-shear correction
        apply_blockage   : apply channel blockage correction (default True)
        """
        v_surface = np.asarray(current_speed_ms, dtype=float)
        rho = self.rho_ref if water_density is None else np.asarray(water_density, dtype=float)

        # Depth-shear velocity correction
        if water_depth_m is not None:
            v_corr = self.depth_velocity_correction(water_depth_m)
            v_hub = v_surface * v_corr
        else:
            v_hub = v_surface

        # Blockage Cp correction
        Cp_eff = self.Cp * self.blockage_correction() if apply_blockage else self.Cp

        # Available kinetic power
        P_avail_w = 0.5 * rho * self.A * Cp_eff * v_hub ** 3 * self.eta
        P_kw = P_avail_w / 1000.0
        P_kw = np.minimum(P_kw, self.P_rated)

        # Cut-in / cut-out
        P_kw = np.where(v_hub < self.v_cut_in, 0.0, P_kw)
        P_kw = np.where(v_hub >= self.v_cut_out, 0.0, P_kw)

        return np.clip(P_kw, 0.0, self.P_rated)

    def power_kw_with_density(self, current_speed_ms, T_C=None, S_psu=None,
                               water_depth_m=None, apply_blockage=True):
        """Convenience: compute density from T/S then call power_kw."""
        rho = self.seawater_density(T_C, S_psu)
        return self.power_kw(current_speed_ms, rho, water_depth_m, apply_blockage)

    def bidirectional_daily_energy_kwh(self, v_ebb, v_flood, T_tide_s,
                                        T_C=None, S_psu=None, water_depth_m=None):
        """
        Daily energy from bidirectional tidal operation [kWh].

        Assumes sinusoidal velocity profiles for ebb and flood tides.
        Average power ≈ 0.5 * P(v_peak) for sinusoidal profile.

        Parameters
        ----------
        v_ebb  : peak ebb current speed [m/s]
        v_flood: peak flood current speed [m/s]
        T_tide_s: tidal period [s]
        """
        rho = self.seawater_density(T_C, S_psu)
        # Each half-period: average power = peak power * 0.5 (sinusoidal)
        # (integral of sin^3 ≈ 4/3π ≈ 0.424 of peak, use 0.42 factor)
        SIN_CUBE_AVG = 0.424
        P_ebb   = float(self.power_kw(v_ebb, rho, water_depth_m)) * SIN_CUBE_AVG
        P_flood = float(self.power_kw(v_flood, rho, water_depth_m)) * SIN_CUBE_AVG
        T_half_h = T_tide_s / 3600.0 / 2.0
        n_cycles_per_day = 86400.0 / T_tide_s
        return (P_ebb + P_flood) * T_half_h * n_cycles_per_day

    def capacity_factor(self, current_speed_ms, water_density=None, water_depth_m=None):
        return self.power_kw(current_speed_ms, water_density, water_depth_m) / self.P_rated
