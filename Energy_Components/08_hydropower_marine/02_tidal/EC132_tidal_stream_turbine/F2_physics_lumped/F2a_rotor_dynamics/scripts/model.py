"""
EC132 -- Tidal Stream Turbine -- F2a Physics-Lumped Rotor Dynamics

0D first-principles model of a horizontal-axis tidal stream turbine. The rotor is
treated as a single lumped inertia driven by the hydrodynamic torque extracted from
a moving water column and braked by the generator/control torque:

    Hydrodynamic power (kinetic flux through swept area):
        P_hydro = 0.5 * rho_water * A * Cp(lambda) * v^3        [W]
        A       = pi * R^2                                      rotor swept area
        lambda  = R * omega / v                                 tip-speed ratio

    Power coefficient curve (BEM-derived, Betz-limited):
        Cp(lambda) = Cp_max * sin^2( pi/2 * lambda / lambda_opt )   for 0<=lambda<=lambda_opt
                   smoothly decaying parabola to 0 at lambda_zero    for lambda>lambda_opt
        Cp is hard-clipped to the Betz limit 16/27 = 0.5926.

    Rotor dynamics ODE (Newton's 2nd law for rotation):
        J * domega/dt = T_hydro - T_gen
        T_hydro = P_hydro / omega        (hydrodynamic torque on the shaft)
        T_gen   = K * omega^2            (below-rated optimal "Kw^2" torque control)
                  P_rated / omega        (above rated, constant-power generator)

    Tidal forcing (semidiurnal sinusoid, ebb+flood symmetric magnitude):
        v(t) = v_mean + v_amp * sin(2*pi*t / T_tide)
        speed magnitude |v| used so both flood and ebb generate.

    Power curve gating (cut-in / rated / cut-out):
        |v| < v_cut_in    -> P = 0   (turbine idles, insufficient flow)
        v_cut_in..v_rated -> P rises with v^3
        v_rated..v_cut_out-> P limited to P_rated (pitch/torque regulated)
        |v| >= v_cut_out  -> P = 0   (feathered, storm protection)

Integrated with scipy.integrate.solve_ivp (LSODA) for the omega state.

Physics guarantees enforced/checked:
  * Cp(lambda) <= Betz limit (16/27) everywhere.
  * Available power scales as v^3 (verified in tests).
  * Energy conservation: integral of P_mech*dt <= integral of P_hydro_avail*dt.
  * Electrical power never exceeds P_rated (rated-power limiting).

References:
    Bahaj, A.S. & Myers, L.E. (2003), "Fundamentals applicable to the utilisation of
        marine current turbines for energy production", Renewable Energy 28(14),
        2205-2211.
    Bahaj, A.S., Molland, A.F., Chaplin, J.R. & Batten, W.M.J. (2007), "Power and
        thrust measurements of marine current turbines under various hydrodynamic flow
        conditions in a cavitation tunnel and a towing tank", Renewable Energy 32(3),
        407-426.
    Fraenkel, P.L. (2002, 2007), "Power from marine currents" & SeaGen project,
        Proc. Inst. Mech. Eng. Part A: J. Power and Energy.
    Seawater density rho = 1025 kg/m3 (S=35 PSU, T~10C) -- Bahaj & Myers (2003),
        IEC TS 62600-200:2013.
"""

import numpy as np
from scipy.integrate import solve_ivp

BETZ_LIMIT = 16.0 / 27.0  # 0.592593


class TidalStreamTurbineF2a:
    """Horizontal-axis tidal stream turbine -- lumped rotor-dynamics ODE model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated_w = u["rated_power"]["value"] * 1000.0   # kW -> W
        self.D = u["rotor_diameter"]["value"]                  # m
        self.R = self.D / 2.0                                  # m
        self.Cp_max = u["Cp_max"]["value"]
        self.lambda_opt = u["lambda_opt"]["value"]
        self.lambda_zero = u["lambda_zero"]["value"]
        self.rho = u["rho_water"]["value"]                     # kg/m3
        self.eta = u["eta_drivetrain"]["value"]
        self.v_cut_in = u["cut_in_speed"]["value"]             # m/s
        self.v_rated = u["rated_speed"]["value"]               # m/s
        self.v_cut_out = u["cut_out_speed"]["value"]           # m/s
        self.J = u["J_rotor"]["value"]                         # kg.m2
        self.omega_rated = u["omega_rated"]["value"]           # rad/s
        self.K_gen = u["K_gen_gain"]["value"]                  # N.m.s2/rad2

        self.A = np.pi * self.R ** 2                           # swept area m2

    # ------------------------------------------------------------------
    # Power coefficient curve (BEM-derived, Betz-limited)
    # ------------------------------------------------------------------
    def cp(self, lam):
        """
        Cp(lambda): rises as sin^2 to Cp_max at lambda_opt, then parabolically
        decays to zero at lambda_zero. Hard-clipped to the Betz limit.
        """
        lam = np.asarray(lam, dtype=float)
        lam = np.maximum(lam, 0.0)

        # Rising branch (0 -> lambda_opt): sin^2 shape, smooth and zero at lam=0
        rising = self.Cp_max * np.sin(0.5 * np.pi * lam / self.lambda_opt) ** 2

        # Falling branch (lambda_opt -> lambda_zero): parabola Cp_max -> 0
        denom = (self.lambda_zero - self.lambda_opt) ** 2
        falling = self.Cp_max * (1.0 - ((lam - self.lambda_opt) / np.sqrt(denom)) ** 2)

        cp_val = np.where(lam <= self.lambda_opt, rising, falling)
        cp_val = np.where(lam >= self.lambda_zero, 0.0, cp_val)
        cp_val = np.clip(cp_val, 0.0, BETZ_LIMIT)
        return cp_val if cp_val.ndim else float(cp_val)

    def tip_speed_ratio(self, omega, v):
        """lambda = R*omega / v (returns 0 for v->0)."""
        v = abs(float(v))
        if v < 1e-9:
            return 0.0
        return self.R * float(omega) / v

    # ------------------------------------------------------------------
    # Power / torque
    # ------------------------------------------------------------------
    def power_available(self, v, rho=None):
        """Raw kinetic flux through the swept area, 0.5*rho*A*v^3 [W]."""
        rho = self.rho if rho is None else float(rho)
        v = abs(float(v))
        return 0.5 * rho * self.A * v ** 3

    def hydro_power(self, omega, v, rho=None):
        """Hydrodynamic mechanical power extracted by the rotor, P=0.5*rho*A*Cp*v^3 [W]."""
        rho = self.rho if rho is None else float(rho)
        lam = self.tip_speed_ratio(omega, v)
        return self.cp(lam) * self.power_available(v, rho)

    def hydro_torque(self, omega, v, rho=None):
        """T_hydro = P_hydro / omega [N.m]. Uses a small floor on omega."""
        omega_eff = max(float(omega), 1e-3)
        return self.hydro_power(omega, v, rho) / omega_eff

    def gen_torque(self, omega):
        """
        Generator/control torque [N.m].
          below rated speed: optimal Kw^2 torque control (tracks Cp_max),
          at/above rated speed: constant-power braking P_rated/omega.
        """
        omega = max(float(omega), 1e-3)
        if omega < self.omega_rated:
            return self.K_gen * omega ** 2
        return self.P_rated_w / omega

    # ------------------------------------------------------------------
    # Electrical power with cut-in / rated / cut-out gating
    # ------------------------------------------------------------------
    def electrical_power_w(self, omega, v, rho=None):
        """Electrical output [W] after drivetrain efficiency, rated limit and gating."""
        speed = abs(float(v))
        if speed < self.v_cut_in or speed >= self.v_cut_out:
            return 0.0
        p_mech = self.hydro_power(omega, v, rho)
        p_elec = self.eta * p_mech
        return float(np.clip(p_elec, 0.0, self.P_rated_w))

    # ------------------------------------------------------------------
    # ODE: J domega/dt = T_hydro - T_gen
    # ------------------------------------------------------------------
    def _rhs(self, t, y, v_func, rho):
        omega = max(y[0], 0.0)
        v = v_func(t)
        T_h = self.hydro_torque(omega, v, rho)
        T_g = self.gen_torque(omega)
        return [(T_h - T_g) / self.J]

    def simulate(self, v_mean=2.0, v_amp=1.0, tidal_period_s=44700.0,
                 duration_s=44700.0, dt=60.0, omega0=None, rho=None,
                 v_func=None):
        """
        Integrate the rotor-dynamics ODE under sinusoidal tidal forcing.

        Args:
            v_mean: mean current speed [m/s]
            v_amp:  semidiurnal amplitude [m/s]
            tidal_period_s: tidal period (semidiurnal ~ 12.42 h = 44712 s)
            duration_s: total simulation time [s]
            dt: output sampling interval [s]
            omega0: initial rotor speed [rad/s] (default ~ optimal for v_mean)
            rho: seawater density [kg/m3] (default design value)
            v_func: optional callable v(t) overriding the sinusoid (uses |v|)

        Returns dict of time-series arrays.
        """
        rho = self.rho if rho is None else float(rho)

        if v_func is None:
            def v_of_t(t):
                return abs(v_mean + v_amp * np.sin(2.0 * np.pi * t / tidal_period_s))
        else:
            def v_of_t(t):
                return abs(float(v_func(t)))

        if omega0 is None:
            # start near optimal TSR for mean speed
            omega0 = self.lambda_opt * max(v_mean, self.v_cut_in) / self.R
        omega0 = max(float(omega0), 1e-3)

        n = max(int(round(duration_s / dt)) + 1, 2)
        t_eval = np.linspace(0.0, duration_s, n)

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [omega0],
            t_eval=t_eval, args=(v_of_t, rho),
            method="LSODA", rtol=1e-6, atol=1e-6, max_step=dt,
        )

        t = sol.t
        omega = np.maximum(sol.y[0], 0.0)
        v = np.array([v_of_t(ti) for ti in t])
        lam = np.array([self.tip_speed_ratio(w, vi) for w, vi in zip(omega, v)])
        cp = self.cp(lam)
        p_avail = np.array([self.power_available(vi, rho) for vi in v])
        p_hydro = np.array([self.hydro_power(w, vi, rho) for w, vi in zip(omega, v)])
        p_elec = np.array([self.electrical_power_w(w, vi, rho) for w, vi in zip(omega, v)])
        rpm = omega * 60.0 / (2.0 * np.pi)

        energy_elec_wh = float(np.trapz(p_elec, t) / 3600.0)
        energy_avail_wh = float(np.trapz(p_avail, t) / 3600.0)
        cf = float(np.mean(p_elec) / self.P_rated_w)

        return {
            "t": t,
            "v": v,
            "omega": omega,
            "rpm": rpm,
            "lambda": lam,
            "cp": cp,
            "power_available_w": p_avail,
            "power_hydro_w": p_hydro,
            "power_electrical_w": p_elec,
            "power_electrical_kw": p_elec / 1000.0,
            "energy_electrical_wh": energy_elec_wh,
            "energy_available_wh": energy_avail_wh,
            "capacity_factor": cf,
        }
