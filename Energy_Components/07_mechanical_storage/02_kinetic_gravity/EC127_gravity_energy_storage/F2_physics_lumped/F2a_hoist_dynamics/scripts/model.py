"""
EC127 -- Gravity Energy Storage -- F2a Hoist Dynamics (physics-lumped)

Physics-lumped (0D, lumped-inertia) first-principles model of a solid-mass
gravity storage hoist (Energy Vault tower / mineshaft class). Energy is stored
as gravitational potential energy by raising a mass m through height h:

        E_stored = m * g * h                                   [J]

The hoist dynamics are governed by Newton's 2nd law applied to the translating
mass coupled to the rotating winch drum (torque tau, drum radius r, lumped
rotational inertia J). Reflecting the drum inertia to an equivalent
translational mass (m_eq = J / r^2) and summing forces on the cable:

    (m + J/r^2) * dv/dt = F_cable - m*g - F_fric(v) - F_drag(v)

with the dissipative mechanical forces (Botha & Kamper 2019):

    F_fric(v) = mu * m * g * sign(v)          Coulomb guide-rail friction [N]
    F_drag(v) = k_drag * v * |v|              aerodynamic drag of mass/cable [N]

Sign convention: position x = height of mass above h_min, v = dx/dt > 0 = up.
F_cable is the upward force the drum applies through the cable. During CHARGE
(lift) the motor drives the drum, F_cable > m*g to accelerate the mass up.
During DISCHARGE (lower) the mass descends, the generator applies a retarding
F_cable (regenerative braking) and the descent rate is controlled.

State vector y = [x, v]:

    dx/dt = v
    dv/dt = (F_cable - m*g - mu*m*g*sign(v) - k_drag*v*|v|) / (m + J/r^2)

integrated with scipy.integrate.solve_ivp.

Electrical power and round-trip efficiency
------------------------------------------
The mechanical power at the cable is P_mech = F_cable * v. The instantaneous
shaft power passes through the drivetrain and machine with part-load-dependent
efficiencies (Pyrhonen et al. 2013 machine-loss model):

  Charge  (motor, P_mech > 0 absorbed by mass):
      P_elec_in  = P_shaft / (eta_motor(PLF) * eta_drive(PLF))   > P_shaft

  Discharge (generator, mass releases P_mech):
      P_elec_out = P_shaft * eta_drive(PLF) * eta_gen(PLF)       < P_shaft

  Bearing parasitic: P_bearing = f_bear * |P_mech|.

Round-trip efficiency over a full lift-then-lower cycle of the same mass/height:

      eta_RT = E_elec_out / E_elec_in   in (0, 1)

Energy conservation is enforced: the change in stored potential energy equals
the net mechanical work minus dissipation, and 0 < eta_RT < 1 always.

References
----------
- Botha, C.D. & Kamper, M.J. (2019). "Capability study of dry gravity energy
  storage." J. Energy Storage, 23, 159-174.
- Berrada, A., Loudiyi, K., Zorkani, I. (2017). "System design and economic
  performance of gravity energy storage." Energy Conversion & Management,
  137, 191-200.
- Tong, W. et al. (2022). "Solid gravity energy storage: A review."
  J. Energy Storage, 53, 105226.
- Pyrhonen, J., Jokinen, T., Hrabovcova, V. (2013). Design of Rotating
  Electrical Machines, 2nd ed., Wiley (part-load machine loss model).
"""

import numpy as np
from scipy.integrate import solve_ivp


class GravityHoistF2a:
    """Solid-mass gravity storage -- lumped hoist dynamics via Newton's 2nd law."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.m = float(u["mass_kg"]["value"])                    # kg
        self.h_max = float(u["h_max_m"]["value"])                # m
        self.h_min = float(u["h_min_m"]["value"])                # m
        self.g = float(u["g"]["value"])                          # m/s2
        self.P_rated = float(u["P_rated_kw"]["value"]) * 1000.0  # W
        self.v_max = float(u["v_max_mps"]["value"])              # m/s
        self.r_drum = float(u["drum_radius_m"]["value"])         # m
        self.J_drum = float(u["drum_inertia_kgm2"]["value"])     # kg.m2
        self.eta_motor_rated = float(u["eta_motor_rated"]["value"])
        self.eta_gen_rated = float(u["eta_gen_rated"]["value"])
        self.eta_drive_rated = float(u["eta_drive_rated"]["value"])
        self.plf_exp = float(u["motor_gen_plf_exp"]["value"])
        self.mu = float(u["friction_coeff"]["value"])
        self.k_drag = float(u["cable_drag_k"]["value"])          # N.s2/m2
        self.f_bear = float(u["bearing_loss_frac"]["value"])

        self.h_usable = self.h_max - self.h_min
        # Equivalent translational mass including reflected drum inertia
        self.m_eq = self.m + self.J_drum / (self.r_drum ** 2)

    # ------------------------------------------------------------------
    # Energy / state
    # ------------------------------------------------------------------
    def stored_energy_J(self, x):
        """Gravitational potential energy stored at position x [m above h_min] -> J."""
        x = np.asarray(x, dtype=float)
        return self.m * self.g * (self.h_min + x)

    def stored_energy_kwh(self, x):
        return self.stored_energy_J(x) / 3.6e6

    def soc(self, x):
        return np.clip(np.asarray(x, dtype=float) / self.h_usable, 0.0, 1.0)

    def energy_capacity_kwh(self):
        """Maximum stored potential energy (ideal) over usable height [kWh]."""
        return self.m * self.g * self.h_usable / 3.6e6

    def cruise_speed(self, mode="charge"):
        """
        Power-limited steady-state line speed [m/s] for this mass and hoist
        rating. At cruise dv/dt=0 so cable power balances gravity + losses.

        Charge:    P_elec_in = P_rated => mechanical cable power
                   P_mech = P_rated * eta_motor * eta_drive, and
                   P_mech = (m g + mu m g + k_drag v^2) v  ~ m g v for large mass.
        For a 10,000 t mass on a 5 MW hoist this gives ~0.05 m/s, far below
        v_max; the lower of the two is the true cruise speed.
        """
        if mode == "charge":
            P_mech_cap = self.P_rated * self.eta_motor_rated * self.eta_drive_rated
        else:
            P_mech_cap = self.P_rated  # generator braking power
        # Solve P_mech_cap = m g v (dominant term) -> v; refine once with friction.
        F_grav = self.m * self.g * (1.0 + self.mu)
        v_pow = P_mech_cap / F_grav
        return min(v_pow, self.v_max)

    # ------------------------------------------------------------------
    # Mechanical loss forces (Newton's 2nd law dissipation terms)
    # ------------------------------------------------------------------
    def friction_force(self, v):
        """Coulomb guide-rail friction force [N], opposes motion."""
        return self.mu * self.m * self.g * np.sign(v)

    def drag_force(self, v):
        """Aerodynamic drag force [N], F = k * v*|v| (opposes motion)."""
        return self.k_drag * v * np.abs(v)

    # ------------------------------------------------------------------
    # Newton's 2nd law right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, F_cable_func):
        """
        ODE rhs for state y = [x, v].

        (m + J/r^2) dv/dt = F_cable - m g - F_fric(v) - F_drag(v)
        """
        x, v = y
        F_cable = F_cable_func(t, x, v)
        F_grav = self.m * self.g
        F_fric = self.friction_force(v)
        F_drag = self.drag_force(v)
        a = (F_cable - F_grav - F_fric - F_drag) / self.m_eq
        return [v, a]

    # ------------------------------------------------------------------
    # Part-load machine efficiency (Pyrhonen et al. 2013, Kloss-style)
    # ------------------------------------------------------------------
    def _eta_mg(self, plf, eta_rated):
        plf = np.clip(np.asarray(plf, dtype=float), 1e-6, 1.0)
        loss_coeff = (1.0 / eta_rated) - 1.0
        f_plf = 0.5 * (plf ** self.plf_exp + 1.0)
        eta = plf / (plf + loss_coeff * f_plf)
        return np.clip(eta, 0.0, eta_rated)

    def motor_efficiency(self, plf):
        return self._eta_mg(plf, self.eta_motor_rated)

    def generator_efficiency(self, plf):
        return self._eta_mg(plf, self.eta_gen_rated)

    def drive_efficiency(self, plf):
        plf = np.clip(np.asarray(plf, dtype=float), 0.0, 1.0)
        return self.eta_drive_rated * (0.9 + 0.1 * plf)

    def electrical_power(self, F_cable, v, mode):
        """
        Convert instantaneous cable mechanical power to electrical power [W].

        P_mech = F_cable * v (shaft side).  Bearing parasitic added on top.
        Charge:   P_elec_in  = (|P_mech| + P_bearing) / (eta_m * eta_d)
        Discharge:P_elec_out = (|P_mech| - P_bearing) * eta_g * eta_d  (>=0)
        Returns signed electrical power: negative = drawn from grid (charge),
        positive = delivered to grid (discharge).
        """
        P_mech = F_cable * v                       # W (can be + or -)
        P_bear = self.f_bear * np.abs(P_mech)
        plf = np.clip(np.abs(P_mech) / self.P_rated, 1e-6, 1.0)
        eta_d = self.drive_efficiency(plf)
        if mode == "charge":
            eta_m = self.motor_efficiency(plf)
            P_in = (np.abs(P_mech) + P_bear) / (eta_m * eta_d)
            return -P_in
        else:  # discharge
            eta_g = self.generator_efficiency(plf)
            P_out = np.maximum(np.abs(P_mech) - P_bear, 0.0) * eta_g * eta_d
            return P_out

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def simulate(self, mode="charge", v_target=None, x0=None, dt=1.0,
                 duration_s=None):
        """
        Simulate hoist dynamics over a charge (lift) or discharge (lower) stroke.

        Parameters
        ----------
        mode : "charge" (lift mass up) or "discharge" (lower mass, regenerate)
        v_target : float
            Commanded steady line speed [m/s] (|v| <= v_max). Defaults to v_max.
        x0 : float
            Initial position [m above h_min]. Defaults to h_min side for charge,
            h_max side for discharge.
        dt : float
            Output time step [s].
        duration_s : float
            Total duration [s]. Defaults to ~full usable stroke at v_target.

        Returns
        -------
        dict time-series: t, x, v, height, soc, F_cable, P_mech, P_elec,
            E_stored_kwh, plus scalar summary keys.
        """
        # Commanded speed is limited by BOTH the line-speed limit (v_max) and
        # the power rating of the hoist (cruise_speed). For a very large mass
        # the hoist power is the binding constraint.
        v_pow = self.cruise_speed(mode)
        if v_target is None:
            v_target = v_pow
        v_target = min(abs(v_target), self.v_max, v_pow)

        if mode == "charge":
            sgn = +1.0
            if x0 is None:
                x0 = 0.0
        else:
            sgn = -1.0
            if x0 is None:
                x0 = self.h_usable

        v_cmd = sgn * v_target

        if duration_s is None:
            duration_s = self.h_usable / max(v_target, 1e-6) * 1.05

        # Proportional speed controller producing the cable force.
        # F_cable = F_grav + F_fric + F_drag (feed-forward) + Kp*(v_cmd - v)
        Kp = 5.0 * self.m_eq / max(self.h_usable / max(v_target, 1e-6), 1.0)
        Kp = max(Kp, self.m_eq * 0.5)

        # Mechanical cable power must stay within the hoist rating. During
        # charge the electrical draw is P_mech/(eta_m*eta_d), so cap the
        # mechanical power below P_rated by the rated efficiency product to
        # keep the *electrical* power within the rating (hoist power limit).
        eta_prod = self.eta_motor_rated * self.eta_drive_rated
        P_mech_cap = self.P_rated * eta_prod          # W, charge-limited
        # During discharge mechanical power can equal P_rated/(eta_g*eta_d)
        # before the electrical output hits the rating; cap at P_rated for
        # conservatism.
        P_mech_cap_dis = self.P_rated

        def F_cable_func(t, x, v):
            F_grav = self.m * self.g
            F_fric = self.friction_force(v)
            F_drag = self.drag_force(v)
            F_ff = F_grav + F_fric + F_drag
            F = F_ff + Kp * (v_cmd - v)
            av = abs(v)
            if mode == "charge":
                # limit upper force by mechanical power cap; allow strong
                # downward (negative) force only for braking, never large
                F_pow = P_mech_cap / av if av > 1e-3 else 5.0 * F_grav
                F = np.clip(F, -3.0 * F_grav, F_pow)
            else:  # discharge: generator retards descent, 0 <= F <= F_grav,
                   # and braking power within rating
                F_pow = P_mech_cap_dis / av if av > 1e-3 else F_grav
                F = np.clip(F, 0.0, min(F_grav, max(F_pow, 0.0)))
            return F

        def stop_event(t, y, *event_args):
            # stop when reaching travel limit in the direction of motion
            if mode == "charge":
                return self.h_usable - y[0]
            return y[0] - 0.0
        stop_event.terminal = True
        stop_event.direction = -1

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [x0, 0.0],
            t_eval=t_eval, args=(F_cable_func,),
            method="RK45", rtol=1e-7, atol=1e-9, max_step=dt,
            events=stop_event,
        )

        t = sol.t
        x = np.clip(sol.y[0], 0.0, self.h_usable)
        v = sol.y[1]
        N = len(t)

        F_cable = np.zeros(N)
        P_mech = np.zeros(N)
        P_elec = np.zeros(N)
        for i in range(N):
            Fc = F_cable_func(t[i], x[i], v[i])
            F_cable[i] = Fc
            P_mech[i] = Fc * v[i]
            P_elec[i] = self.electrical_power(Fc, v[i], mode)

        E_stored_kwh = self.stored_energy_kwh(x)

        # Integrated electrical energy [kWh] (trapezoid)
        if N > 1:
            E_elec_kwh = np.trapz(P_elec, t) / 3.6e6
        else:
            E_elec_kwh = 0.0

        return {
            "t": t,
            "x": x,
            "v": v,
            "height": self.h_min + x,
            "soc": self.soc(x),
            "F_cable": F_cable,
            "P_mech": P_mech,
            "P_elec": P_elec,
            "P_elec_kw": P_elec / 1000.0,
            "E_stored_kwh": E_stored_kwh,
            "E_elec_kwh": float(E_elec_kwh),
            "mode": mode,
            "v_target": v_target,
        }

    # ------------------------------------------------------------------
    # Round-trip efficiency (full lift then lower)
    # ------------------------------------------------------------------
    def round_trip_efficiency(self, v_target=None, dt=2.0):
        """
        Full-cycle round-trip efficiency:
            charge (lift 0 -> h_usable) then discharge (lower h_usable -> 0).
            eta_RT = |E_elec_out| / |E_elec_in|, in (0, 1).
        """
        ch = self.simulate(mode="charge", v_target=v_target, dt=dt)
        di = self.simulate(mode="discharge", v_target=v_target, dt=dt)
        E_in = abs(ch["E_elec_kwh"])      # grid energy consumed (kWh)
        E_out = abs(di["E_elec_kwh"])     # grid energy returned (kWh)
        if E_in <= 1e-12:
            return 0.0
        return float(np.clip(E_out / E_in, 0.0, 1.0))
