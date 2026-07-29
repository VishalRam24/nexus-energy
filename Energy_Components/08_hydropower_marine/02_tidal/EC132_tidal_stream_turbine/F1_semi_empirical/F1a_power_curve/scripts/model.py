"""
EC132 — Tidal Stream Turbine — F1a Power Curve Model

P = 0.5 * rho_water * A * Cp(v) * v^3 * eta_drivetrain  [W]

Fluid-dynamically analogous to a wind turbine (EC062) but:
  - rho_water = 1025 kg/m3 (vs 1.225 kg/m3 for air → ~840x denser)
  - Much lower flow speeds (1-4 m/s vs 3-25 m/s for wind)
  - Cp limited to ~0.35–0.45 (same Betz physics)
  - Bidirectional capability (ebb and flood generation)

Power curve regions:
  v < v_cut_in          : P = 0  (insufficient flow)
  v_cut_in <= v < v_rated: P = 0.5 * rho * A * Cp * v^3 * eta  (rising)
  v_rated <= v < v_cut_out: P = P_rated (constant, pitch regulated)
  v >= v_cut_out         : P = 0  (storm protection, turbine feathered)

Reference:
    Fraenkel (2002), "Power from Marine Currents", Proc. Inst. Mech. Eng. A.
    IEC TS 62600-200:2013, Marine energy - Tidal energy resource assessment.
    EMEC (2009), Assessment of Tidal Energy Resource, European Marine Energy Centre.
"""

import numpy as np


class TidalStreamTurbineF1a:
    """Horizontal-axis tidal stream turbine — power curve with density correction."""

    def __init__(self, params: dict):
        t = params["turbine"]
        self.P_rated = t["rated_power"]["value"]          # kW
        self.D = t["rotor_diameter"]["value"]             # m
        self.Cp = t["Cp_rated"]["value"]                  # peak power coefficient
        self.v_cut_in = t["cut_in_speed"]["value"]        # m/s
        self.v_rated = t["rated_speed"]["value"]          # m/s
        self.v_cut_out = t["cut_out_speed"]["value"]      # m/s
        self.eta = t["eta_drivetrain"]["value"]
        self.rho_ref = t["rho_water"]["value"]            # kg/m3

        self.A = np.pi * (self.D / 2.0) ** 2             # rotor swept area [m2]

    def power_kw(self, current_speed_ms, water_density=None):
        """
        Electrical power output in kW.

        Args:
            current_speed_ms: Tidal current speed [m/s] (scalar or array)
            water_density: Seawater density [kg/m3] (default: design value)
        """
        v = np.asarray(current_speed_ms, dtype=float)
        rho = self.rho_ref if water_density is None else float(water_density)

        # Available tidal power [W] (pre-rated)
        P_avail_w = 0.5 * rho * self.A * self.Cp * v ** 3 * self.eta

        # Apply power curve regions
        P_kw = P_avail_w / 1000.0

        # Flatten at rated power
        P_kw = np.minimum(P_kw, self.P_rated)

        # Cut-in: no power below minimum current
        P_kw = np.where(v < self.v_cut_in, 0.0, P_kw)

        # Cut-out: no power above maximum current (storm protection)
        P_kw = np.where(v >= self.v_cut_out, 0.0, P_kw)

        return np.clip(P_kw, 0.0, self.P_rated)

    def capacity_factor(self, current_speed_ms, water_density=None):
        """Capacity factor = P / P_rated."""
        return self.power_kw(current_speed_ms, water_density) / self.P_rated

    def power_coefficient(self, current_speed_ms, water_density=None):
        """
        Operational Cp = P_electrical / P_available_kinetic.
        Note: below cut-in and above cut-out, P=0 so Cp=0.
        """
        v = np.asarray(current_speed_ms, dtype=float)
        rho = self.rho_ref if water_density is None else float(water_density)
        P_kw = self.power_kw(v, rho)
        P_avail_kw = 0.5 * rho * self.A * v ** 3 / 1000.0
        return np.where(P_avail_kw > 0, P_kw / P_avail_kw, 0.0)

    def density_correction_factor(self, water_density):
        """Power ratio relative to reference density."""
        return float(water_density) / self.rho_ref
