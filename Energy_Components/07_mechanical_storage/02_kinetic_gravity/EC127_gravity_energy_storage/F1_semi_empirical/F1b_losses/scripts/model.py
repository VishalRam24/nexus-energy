"""
EC127 — Gravity Energy Storage — F1b Losses Model

Extends F1a (ideal potential energy) by adding:

1. Speed-dependent mechanical losses:
   a. Coulomb (static/sliding) friction at guide rails:
      P_friction = mu * m * g * v   [W]
      - Proportional to velocity and mass weight
   b. Aerodynamic drag on cable/mass:
      P_drag = k_drag * v^2   [W]
      - Quadratic with velocity (from drag equation)
   c. Bearing/seal losses:
      P_bearing = f_bear * P_mech   [W]
      - Proportional to instantaneous mechanical power

   Total mechanical loss: P_mech_loss = P_friction + P_drag + P_bearing

2. Partial-load motor/generator efficiency:
   - Full-load efficiency = eta_rated (from F1a)
   - Part-load derating via simplified motor loss model:
       eta(PLF) = PLF / (PLF + (1 - eta_rated) * (PLF^2 + 1/PLF_opt))
   Simplified form used here (Kloss-like for induction machines):
       losses = fixed_loss + variable_loss * PLF^n
       where PLF = P_cmd / P_rated
   Final form: eta_mg(PLF) = 1 - (1 - eta_rated) * (1 + (PLF - 1)^plf_exp)
   Clamped to [0, eta_rated].

3. Effective electrical power:
   Charge (lifting):
     P_elec_in = (P_gravity + P_mech_loss) / (eta_motor(PLF) * eta_drive(PLF))
   Discharge (lowering):
     P_elec_out = (P_gravity - P_mech_loss) * eta_drive(PLF) * eta_gen(PLF)

4. Round-trip efficiency with losses:
   eta_RT = P_elec_out / P_elec_in (at same velocity/load)

References:
    Botha, C.D. & Kamper, M.J. (2019). J. Energy Storage, 23, 159-174.
    Berrada, A. et al. (2017). Energy Conversion and Management, 137, 191-200.
    Pyrhonen, J. et al. (2013). Design of Rotating Electrical Machines. Wiley.
"""

import numpy as np


class GravityF1b:
    """Gravity Energy Storage — losses model with friction, drag, and part-load efficiency."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.m = u["mass_kg"]["value"]                        # kg
        self.h_max = u["h_max_m"]["value"]                    # m
        self.h_min = u["h_min_m"]["value"]                    # m
        self.g = u["g"]["value"]                              # m/s2
        self.P_rated = u["P_rated_kw"]["value"] * 1000.0      # W
        self.eta_motor_rated = u["eta_motor_rated"]["value"]
        self.eta_gen_rated = u["eta_gen_rated"]["value"]
        self.eta_drive_rated = u["eta_drive_rated"]["value"]
        self.plf_exp = u["motor_gen_plf_exp"]["value"]
        self.mu = u["friction_coeff"]["value"]                 # Coulomb friction
        self.k_drag = u["cable_drag_k"]["value"]               # W/(m/s)^2
        self.f_bear = u["bearing_loss_frac"]["value"]          # fraction
        self.v_max = u["v_max_mps"]["value"]                   # m/s

        self._h_usable = self.h_max - self.h_min
        # P_grav at max v and rated height (for reference)
        self._P_grav_rated = self.m * self.g * self.v_max      # W

    # ------------------------------------------------------------------
    # Mechanical state
    # ------------------------------------------------------------------

    def height(self, soc):
        """Mass height [m] at given SOC."""
        s = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        return self.h_min + s * self._h_usable

    def soc_from_height(self, h):
        h = np.asarray(h, dtype=float)
        return np.clip((h - self.h_min) / self._h_usable, 0.0, 1.0)

    def potential_energy_kwh(self, soc):
        """Stored potential energy [kWh]."""
        h = self.height(soc)
        return self.m * self.g * h / 3.6e6

    # ------------------------------------------------------------------
    # Speed-dependent mechanical losses
    # ------------------------------------------------------------------

    def friction_loss_kw(self, v_mps):
        """
        Coulomb friction loss [kW]:
            P_friction = mu * m * g * v
        Linear with velocity — represents guide rail / rope friction.

        Args:
            v_mps: Lift/lower velocity [m/s]
        """
        v = np.asarray(v_mps, dtype=float)
        return self.mu * self.m * self.g * np.abs(v) / 1000.0

    def drag_loss_kw(self, v_mps):
        """
        Aerodynamic drag loss [kW]:
            P_drag = k_drag * v^2
        Quadratic with velocity — cable/mass drag through air shaft.

        Args:
            v_mps: Velocity [m/s]
        """
        v = np.asarray(v_mps, dtype=float)
        return self.k_drag * v ** 2 / 1000.0

    def bearing_loss_kw(self, P_mech_kw):
        """
        Bearing and seal loss [kW]:
            P_bearing = f_bear * |P_mech|
        Proportional to mechanical power being transmitted.

        Args:
            P_mech_kw: Mechanical power magnitude [kW]
        """
        return self.f_bear * np.abs(np.asarray(P_mech_kw, dtype=float))

    def total_mechanical_loss_kw(self, v_mps, P_mech_kw=None):
        """
        Total mechanical losses [kW] = friction + drag + bearing.

        Args:
            v_mps:      Velocity [m/s]
            P_mech_kw:  Mechanical power [kW] (for bearing loss; if None, estimated)
        """
        v = np.asarray(v_mps, dtype=float)
        P_f = self.friction_loss_kw(v)
        P_d = self.drag_loss_kw(v)
        if P_mech_kw is None:
            P_mech_kw = self.m * self.g * np.abs(v) / 1000.0
        P_b = self.bearing_loss_kw(P_mech_kw)
        return P_f + P_d + P_b

    def mechanical_loss_breakdown(self, v_mps):
        """
        Dict of loss components [kW] at given velocity.

        Returns friction_kw, drag_kw, bearing_kw, total_kw.
        """
        v = np.asarray(v_mps, dtype=float)
        P_mech = self.m * self.g * np.abs(v) / 1000.0
        P_f = self.friction_loss_kw(v)
        P_d = self.drag_loss_kw(v)
        P_b = self.bearing_loss_kw(P_mech)
        return {
            "friction_kw": P_f,
            "drag_kw": P_d,
            "bearing_kw": P_b,
            "total_kw": P_f + P_d + P_b,
        }

    # ------------------------------------------------------------------
    # Part-load motor/generator efficiency
    # ------------------------------------------------------------------

    def _eta_mg_from_plf(self, plf, eta_rated):
        """
        Motor/generator efficiency at part-load factor PLF.

        Model: losses = fixed + variable * PLF^2, normalized so eta(1)=eta_rated.
        Simplified form: eta(PLF) = PLF * eta_rated / (PLF + (1 - eta_rated) * (1 + (PLF-1)^n))
        Clamped to [0, eta_rated].

        Args:
            plf:       Part-load factor [0-1]  = P / P_rated
            eta_rated: Rated (full-load) efficiency
        """
        plf = np.clip(np.asarray(plf, dtype=float), 1e-6, 1.0)
        # At PLF=0: eta→0; at PLF=1: eta=eta_rated
        # Loss model: relative losses = (1-eta_rated) * (PLF^n + (1-PLF)^n) / some normalization
        # Simplified: eta(PLF) = 1 - (1-eta_rated)*(1 + (PLF-1)**n)  won't work without care
        # Use standard Kloss-style:
        # eta = PLF / (PLF + (1/eta_rated - 1) * f(PLF))
        # where f(PLF) = 0.5*(PLF^2 + 1) (fixed + variable loss)
        loss_coeff = (1.0 / eta_rated) - 1.0          # = (1-eta_rated)/eta_rated
        f_plf = 0.5 * (plf ** self.plf_exp + 1.0)
        eta = plf / (plf + loss_coeff * f_plf)
        return np.clip(eta, 0.0, eta_rated)

    def motor_efficiency(self, plf):
        """Motor efficiency at part-load factor PLF [0-1]."""
        return self._eta_mg_from_plf(plf, self.eta_motor_rated)

    def generator_efficiency(self, plf):
        """Generator efficiency at part-load factor PLF [0-1]."""
        return self._eta_mg_from_plf(plf, self.eta_gen_rated)

    def drive_efficiency(self, plf):
        """
        Drivetrain efficiency at PLF.
        Simple linear derating: eta_drive(PLF) = eta_drive_rated * (0.9 + 0.1*PLF)
        (Minor derating at part-load — cable/gearbox losses relatively constant)
        """
        plf = np.clip(np.asarray(plf, dtype=float), 0.0, 1.0)
        return self.eta_drive_rated * (0.9 + 0.1 * plf)

    # ------------------------------------------------------------------
    # Velocity from power command
    # ------------------------------------------------------------------

    def velocity_from_power(self, P_mech_kw):
        """
        Lift/lower velocity [m/s] required to deliver P_mech [kW].
        v = P_mech / (m * g)   — simple inverse of gravitational power
        Clamped to v_max.
        """
        P_W = np.asarray(P_mech_kw, dtype=float) * 1000.0
        v = np.abs(P_W) / (self.m * self.g)
        return np.clip(v, 0.0, self.v_max)

    # ------------------------------------------------------------------
    # Actual electrical power with losses
    # ------------------------------------------------------------------

    def charge_power(self, v_mps, plf=None):
        """
        Electrical input power [kW] to lift mass at velocity v.

        P_elec = (P_gravity + P_mech_loss) / (eta_motor(PLF) * eta_drive(PLF))

        Args:
            v_mps: Lift velocity [m/s]
            plf:   Part-load factor [0-1] (if None, computed from P_grav / P_rated)
        """
        v = np.asarray(v_mps, dtype=float)
        P_grav_kw = self.m * self.g * np.abs(v) / 1000.0
        P_loss_kw = self.total_mechanical_loss_kw(v, P_grav_kw)
        P_shaft_kw = P_grav_kw + P_loss_kw

        if plf is None:
            plf = np.clip(P_shaft_kw * 1000.0 / self.P_rated, 1e-6, 1.0)

        eta_m = self.motor_efficiency(plf)
        eta_d = self.drive_efficiency(plf)
        return np.where(v > 0, P_shaft_kw / (eta_m * eta_d), 0.0)

    def discharge_power(self, v_mps, plf=None):
        """
        Electrical output power [kW] when lowering mass at velocity v.

        P_elec = (P_gravity - P_mech_loss) * eta_drive(PLF) * eta_gen(PLF)

        Args:
            v_mps: Lower velocity [m/s]
            plf:   Part-load factor [0-1]
        """
        v = np.asarray(v_mps, dtype=float)
        P_grav_kw = self.m * self.g * np.abs(v) / 1000.0
        P_loss_kw = self.total_mechanical_loss_kw(v, P_grav_kw)
        P_shaft_kw = np.maximum(P_grav_kw - P_loss_kw, 0.0)

        if plf is None:
            plf = np.clip(P_shaft_kw * 1000.0 / self.P_rated, 1e-6, 1.0)

        eta_g = self.generator_efficiency(plf)
        eta_d = self.drive_efficiency(plf)
        return np.where(v > 0, P_shaft_kw * eta_g * eta_d, 0.0)

    # ------------------------------------------------------------------
    # Round-trip efficiency with losses
    # ------------------------------------------------------------------

    def round_trip_efficiency(self, v_mps, plf=None):
        """
        Round-trip efficiency at given velocity:
            eta_RT = P_elec_out / P_elec_in
        """
        P_in = self.charge_power(v_mps, plf)
        P_out = self.discharge_power(v_mps, plf)
        P_in_safe = np.where(P_in > 1e-9, P_in, np.ones_like(P_in))
        return np.where(P_in > 1e-9, np.clip(P_out / P_in_safe, 0.0, 1.0), 0.0)

    def efficiency(self, v_mps, mode="discharge", plf=None):
        """One-way efficiency (charge or discharge) at given velocity."""
        v = np.asarray(v_mps, dtype=float)
        P_grav_kw = self.m * self.g * np.abs(v) / 1000.0
        P_loss_kw = self.total_mechanical_loss_kw(v, P_grav_kw)

        if plf is None:
            P_shaft_kw = P_grav_kw + P_loss_kw if mode == "charge" else np.maximum(P_grav_kw - P_loss_kw, 0.0)
            plf = np.clip(P_shaft_kw * 1000.0 / self.P_rated, 1e-6, 1.0)

        if mode == "discharge":
            eta_g = self.generator_efficiency(plf)
            eta_d = self.drive_efficiency(plf)
            P_out = np.maximum(P_grav_kw - P_loss_kw, 0.0) * eta_g * eta_d
            P_ref = np.where(P_grav_kw > 1e-9, P_grav_kw, 1.0)
            return np.where(P_grav_kw > 1e-9, np.clip(P_out / P_ref, 0.0, 1.0), 0.0)
        else:
            eta_m = self.motor_efficiency(plf)
            eta_d = self.drive_efficiency(plf)
            P_shaft_kw = P_grav_kw + P_loss_kw
            P_in = P_shaft_kw / (eta_m * eta_d)
            return np.where(P_in > 1e-9, np.clip(P_grav_kw / P_in, 0.0, 1.0), 0.0)

    # ------------------------------------------------------------------
    # Energy capacity
    # ------------------------------------------------------------------

    def energy_capacity_kwh(self):
        """Maximum usable electrical energy [kWh] (ideal, from potential energy)."""
        return self.m * self.g * self._h_usable / 3.6e6
